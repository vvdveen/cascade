import sys
import logging
from pathlib import Path

# Add current directory to path to allow importing cascade
sys.path.append(str(Path.cwd()))

from cascade.config import FuzzerConfig, PICORV32_CONFIG
from cascade.generator.intermediate import IntermediateProgramGenerator
from cascade.generator.ultimate import UltimateProgramGenerator, UltimateProgram
from cascade.execution.iss_runner import ISSRunner
from cascade.execution.rtl_runner import RTLRunner
from cascade.reduction.reducer import Reducer
from cascade.execution.elf_writer import ELFWriter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("manual_reduce")

def reduce_bug(seed, bug_id):
    logger.info(f"Reducing bug {bug_id} with seed {seed}")
    
    # Config
    config = FuzzerConfig(
        cpu=PICORV32_CONFIG,
        output_dir=Path("./output"),
        spike_path=Path("/opt/riscv/bin/spike"),
        rtl_model_path=Path("deps/picorv32"),
        seed=seed
    )
    
    # Generators and Runners
    inter_gen = IntermediateProgramGenerator(config)
    ultimate_gen = UltimateProgramGenerator(config)
    iss_runner = ISSRunner(config)
    rtl_runner = RTLRunner(config)
    reducer = Reducer(config, rtl_runner, iss_runner)
    
    # 1. Generate Intermediate
    logger.info("Generating intermediate program...")
    intermediate = inter_gen.generate(seed=seed)
    
    # 2. Run on ISS
    logger.info("Running on ISS...")
    iss_result = iss_runner.run(intermediate, collect_feedback=True)
    if not iss_result.success:
        logger.error(f"ISS run failed: {iss_result.error_message}")
        return

    # 3. Generate Ultimate
    logger.info("Generating ultimate program...")
    ultimate = ultimate_gen.generate(intermediate, iss_result.feedback)
    
    # 4. Verify on RTL
    logger.info("Verifying on RTL...")
    rtl_result = rtl_runner.run(ultimate)
    if not rtl_result.bug_detected:
        logger.error(f"Bug NOT detected on RTL verification! Cannot reduce. Cycle count: {rtl_result.cycle_count}")
        return

    # 5. Reduce
    logger.info("Reducing...")
    reduction = reducer.reduce(ultimate, iss_result.feedback)
    
    # 6. Save
    output_path = Path(f"output/reduced_bugs/{bug_id}/reduced.elf")
    logger.info(f"Saving to {output_path}")
    reducer.save_reduced(reduction, output_path)
    
    # Save assembly
    asm_path = output_path.with_suffix('.S')
    with open(asm_path, "w") as f:
        f.write(_format_program_asm(reduction.reduced_program))
        logger.info(f"Saved assembly to {asm_path}")

def _format_program_asm(program: UltimateProgram) -> str:
    """Format ultimate program as simple assembly listing."""
    lines = []
    for block in sorted(program.blocks, key=lambda b: b.start_addr):
        lines.append(f"# block {block.block_id} @ 0x{block.start_addr:08x}")
        pc = block.start_addr
        for instr in block.instructions:
            lines.append(f"0x{pc:08x}: {instr.to_asm()}")
            pc += 4
        if block.terminator:
            lines.append(f"0x{pc:08x}: {block.terminator.to_asm()}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"

if __name__ == "__main__":
    bugs = [
        (657777256, "bug_20260117_171416_81"),
        # (3526151838, "bug_20260117_171418_32"),
        # (4168765291, "bug_20260117_171418_59")
    ]
    
    for seed, bug_id in bugs:
        try:
            reduce_bug(seed, bug_id)
        except Exception as e:
            logger.error(f"Failed to reduce {bug_id}: {e}")
            import traceback
            traceback.print_exc()
