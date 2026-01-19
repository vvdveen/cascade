#!/bin/bash
#
# Setup script for Cascade dependencies
#
# This script installs:
# - Spike (RISC-V ISS)
# - Verilator (RTL simulator)
# - PicoRV32 (test target)
# - Kronos (test target)
#
set -euo pipefail

INSTALL_PREFIX="${INSTALL_PREFIX:-$HOME/.local/riscv}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BUILD_DIR="${PROJECT_DIR}/deps"
OS_NAME="$(uname -s)"
ARCH_NAME="$(uname -m)"
SUDO_CMD="sudo"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

ensure_install_prefix_writable() {
    if [ -d "$INSTALL_PREFIX" ] && [ -w "$INSTALL_PREFIX" ]; then
        SUDO_CMD=""
        return 0
    fi

    if [ "$OS_NAME" = "Darwin" ]; then
        if [ ! -d "$INSTALL_PREFIX" ]; then
            mkdir -p "$INSTALL_PREFIX" 2>/dev/null || true
        fi
        if [ -d "$INSTALL_PREFIX" ] && [ -w "$INSTALL_PREFIX" ]; then
            SUDO_CMD=""
            return 0
        fi
        if command -v sudo &> /dev/null; then
            if sudo -n true &> /dev/null; then
                SUDO_CMD="sudo"
                return 0
            fi
        fi
        log_error "INSTALL_PREFIX is not writable: $INSTALL_PREFIX"
        log_info "Re-run with sudo or set INSTALL_PREFIX to a writable path (e.g., \$HOME/.local/riscv)."
        exit 1
    fi
}

run_make_install() {
    if [ -n "$SUDO_CMD" ]; then
        $SUDO_CMD make install
    else
        make install
    fi
}

# Check for required tools
check_dependencies() {
    log_info "Checking dependencies..."

    local missing=()

    for cmd in git make g++ autoconf automake pkg-config; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ "$OS_NAME" = "Darwin" ]; then
        if ! command -v glibtoolize &> /dev/null && ! command -v libtoolize &> /dev/null; then
            missing+=("glibtoolize (brew libtool)")
        fi
    else
        # libtool can be either 'libtool' or 'libtoolize' depending on distro
        if ! command -v libtool &> /dev/null && ! command -v libtoolize &> /dev/null; then
            missing+=("libtool")
        fi
    fi

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        if [ "$OS_NAME" = "Darwin" ]; then
            log_info "Install with: brew install autoconf automake libtool pkg-config"
        else
            log_info "Install with: sudo apt-get install git build-essential autoconf automake libtool pkg-config"
        fi
        exit 1
    fi

    log_info "All required tools found"
}

get_num_cores() {
    if command -v nproc &> /dev/null; then
        nproc
    elif [ "$OS_NAME" = "Darwin" ]; then
        sysctl -n hw.ncpu
    else
        echo 1
    fi
}

# Install system dependencies (Debian/Ubuntu)
install_system_deps() {
    log_info "Installing system dependencies..."

    if [ "$OS_NAME" = "Darwin" ]; then
        if ! command -v brew &> /dev/null; then
            log_error "Homebrew not found. Install from https://brew.sh/ and re-run."
            exit 1
        fi

        log_info "Using Homebrew for macOS dependencies..."
        brew install autoconf automake libtool pkg-config \
            boost dtc bison flex ccache gperftools
    elif command -v apt-get &> /dev/null; then
        sudo apt-get update
        sudo apt-get install -y \
            git build-essential autoconf automake libtool \
            device-tree-compiler libboost-all-dev \
            python3 python3-pip \
            flex bison ccache libgoogle-perftools-dev numactl perl-doc \
            libfl2 libfl-dev zlib1g zlib1g-dev
    elif command -v dnf &> /dev/null; then
        sudo dnf install -y \
            git gcc g++ make autoconf automake libtool \
            dtc boost-devel \
            python3 python3-pip \
            flex bison ccache gperftools-libs numactl \
            flex-devel zlib-devel
    else
        log_warn "Unsupported package manager, please install dependencies manually"
    fi
}

# Install Spike (RISC-V ISS)
install_spike() {
    log_info "Installing Spike ISS..."
    ensure_install_prefix_writable

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ -d "riscv-isa-sim" ]; then
        log_info "Spike source already exists, updating..."
        cd riscv-isa-sim
        git fetch origin
    else
        log_info "Cloning Spike..."
        git clone https://github.com/riscv-software-src/riscv-isa-sim.git
        cd riscv-isa-sim
    fi

    # Use specific version from paper
    git checkout fcbdbe79 2>/dev/null || git checkout master

    log_info "Building Spike..."
    mkdir -p build && cd build
    local extra_config=()
    if [ "$OS_NAME" = "Darwin" ] && command -v brew &> /dev/null; then
        extra_config+=(--with-boost="$(brew --prefix boost)")
    fi

    if [ "$OS_NAME" = "Darwin" ]; then
        local toolize_cmd="glibtoolize"
        if ! command -v "$toolize_cmd" &> /dev/null; then
            toolize_cmd="libtoolize"
        fi
        if command -v "$toolize_cmd" &> /dev/null; then
            "$toolize_cmd" --copy --force > /dev/null 2>&1 || true
        fi
    fi

    ../configure --prefix="$INSTALL_PREFIX" "${extra_config[@]}"
    make -j"$(get_num_cores)"
    run_make_install

    log_info "Spike installed to $INSTALL_PREFIX"
}

# Install Verilator
install_verilator() {
    log_info "Installing Verilator..."
    ensure_install_prefix_writable

    # Check if already installed and version >= 5
    if command -v verilator &> /dev/null; then
        local version
        version=$(verilator --version 2>/dev/null | awk 'match($0, /[0-9]+\.[0-9]+/) { print substr($0, RSTART, RLENGTH); exit }')
        if [ -z "$version" ]; then
            log_error "Failed to parse Verilator version."
            return 1
        fi
        local major=$(echo "$version" | cut -d. -f1)
        if [ "$major" -ge 5 ]; then
            log_info "Verilator $version already installed"
            return 0
        fi
    fi

    if [ "$OS_NAME" = "Darwin" ] && command -v brew &> /dev/null; then
        log_info "Installing Verilator via Homebrew..."
        brew install verilator
        return 0
    fi

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ -d "verilator" ]; then
        log_info "Verilator source already exists, updating..."
        cd verilator
        git fetch origin
    else
        log_info "Cloning Verilator..."
        git clone https://github.com/verilator/verilator.git
        cd verilator
    fi

    # Use version 5.x
    git checkout v5.005 2>/dev/null || git checkout master

    log_info "Building Verilator..."
    if [ "$OS_NAME" = "Darwin" ] && command -v brew &> /dev/null; then
        export PATH="$(brew --prefix bison)/bin:$PATH"
    fi
    autoconf
    ./configure --prefix="$INSTALL_PREFIX"
    make -j"$(get_num_cores)"
    run_make_install

    log_info "Verilator installed to $INSTALL_PREFIX"
}

# Install PicoRV32 (test target)
install_picorv32() {
    log_info "Installing PicoRV32..."

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ -d "picorv32" ]; then
        log_info "PicoRV32 source already exists"
    else
        log_info "Cloning PicoRV32..."
        git clone https://github.com/YosysHQ/picorv32.git
        cd picorv32
        git checkout f00a88c 2>/dev/null || git checkout master
    fi

    log_info "PicoRV32 installed to $BUILD_DIR/picorv32"
}

ensure_riscv32_toolchain() {
    local toolchain_dir="$BUILD_DIR/toolchains/riscv32"
    local bin_dir="$toolchain_dir/bin"
    if [ -x "$bin_dir/riscv32-unknown-elf-gcc" ] && \
       [ -x "$bin_dir/riscv32-unknown-elf-objdump" ] && \
       [ -x "$bin_dir/riscv32-unknown-elf-objcopy" ]; then
        echo "$toolchain_dir"
        return 0
    fi

    mkdir -p "$bin_dir"

    if command -v riscv32-unknown-elf-gcc &> /dev/null; then
        ln -sf "$(command -v riscv32-unknown-elf-gcc)" "$bin_dir/riscv32-unknown-elf-gcc"
        ln -sf "$(command -v riscv32-unknown-elf-objdump)" "$bin_dir/riscv32-unknown-elf-objdump"
        ln -sf "$(command -v riscv32-unknown-elf-objcopy)" "$bin_dir/riscv32-unknown-elf-objcopy"
        echo "$toolchain_dir"
        return 0
    fi

    if command -v riscv64-unknown-elf-gcc &> /dev/null; then
        ln -sf "$(command -v riscv64-unknown-elf-gcc)" "$bin_dir/riscv32-unknown-elf-gcc"
        ln -sf "$(command -v riscv64-unknown-elf-objdump)" "$bin_dir/riscv32-unknown-elf-objdump"
        ln -sf "$(command -v riscv64-unknown-elf-objcopy)" "$bin_dir/riscv32-unknown-elf-objcopy"
        echo "$toolchain_dir"
        return 0
    fi

    log_warn "RISC-V toolchain not found (riscv32-unknown-elf-* or riscv64-unknown-elf-*)."
    echo ""
    return 1
}

apply_kronos_patches() {
    local kronos_dir="$1"
    local patch_dir="$PROJECT_DIR/scripts/patches/kronos"
    if [ ! -d "$patch_dir" ]; then
        return 0
    fi

    for patch in "$patch_dir"/*.patch; do
        [ -f "$patch" ] || continue
        if git -C "$kronos_dir" apply --check "$patch" &> /dev/null; then
            log_info "Applying Kronos patch: $(basename "$patch")"
            git -C "$kronos_dir" apply "$patch"
        else
            log_info "Kronos patch already applied: $(basename "$patch")"
        fi
    done
}

try_make_targets() {
    local dir="$1"
    shift
    local targets=("$@")
    local target
    for target in "${targets[@]}"; do
        log_info "Attempting build target '$target' in $dir"
        set +e
        make -C "$dir" "$target" -j"$(get_num_cores)"
        status=$?
        set -e
        if [ "$status" -eq 0 ]; then
            return 0
        fi
    done
    return 1
}

# Install Kronos (test target)
install_kronos() {
    log_info "Installing Kronos..."

    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"

    if [ -d "kronos" ]; then
        log_info "Kronos source already exists"
    else
        log_info "Cloning Kronos..."
        git clone https://github.com/SonalPinto/kronos.git
        cd kronos
        git checkout 13678d4 2>/dev/null || git checkout master
    fi

    apply_kronos_patches "$BUILD_DIR/kronos"

    local riscv_tc
    riscv_tc="$(ensure_riscv32_toolchain || true)"

    mkdir -p "$BUILD_DIR/kronos/build"
    log_info "Configuring Kronos with CMake..."
    if [ -n "$riscv_tc" ]; then
        RISCV_TOOLCHAIN_DIR="$riscv_tc" cmake -S "$BUILD_DIR/kronos" -B "$BUILD_DIR/kronos/build"
    else
        cmake -S "$BUILD_DIR/kronos" -B "$BUILD_DIR/kronos/build"
    fi

    log_info "Building Kronos compliance simulator..."
    if [ -n "$riscv_tc" ]; then
        RISCV_TOOLCHAIN_DIR="$riscv_tc" cmake --build "$BUILD_DIR/kronos/build" --target kronos_compliance -j"$(get_num_cores)" || \
            log_warn "Kronos build failed; check CMake output in $BUILD_DIR/kronos/build"
    else
        cmake --build "$BUILD_DIR/kronos/build" --target kronos_compliance -j"$(get_num_cores)" || \
            log_warn "Kronos build failed; check CMake output in $BUILD_DIR/kronos/build"
    fi

    log_info "Kronos installed to $BUILD_DIR/kronos"
}

# Install Python dependencies
install_python_deps() {
    log_info "Installing Python dependencies..."

    cd "$PROJECT_DIR"

    # Create virtual environment if it doesn't exist
    if [ ! -d ".venv" ]; then
        log_info "Creating virtual environment..."
        python3 -m venv .venv
    fi

    # Activate and install
    source .venv/bin/activate
    pip install -e ".[dev]"

    log_info "Python dependencies installed in .venv"
    log_info "Activate with: source .venv/bin/activate"
}

# Create configuration file
create_config() {
    log_info "Creating default configuration..."

    mkdir -p "$PROJECT_DIR/configs"
    cat > "$PROJECT_DIR/configs/picorv32.yaml" << EOF
# Cascade configuration for PicoRV32
cpu:
  name: picorv32
  xlen: 32
  extensions:
    - I
    - M

memory:
  code_start: 0x00000000
  code_size: 0x10000
  data_start: 0x00010000
  data_size: 0x10000

execution:
  iss_timeout: 10000
  rtl_timeout: 100000
  spike_path: ${INSTALL_PREFIX}/bin/spike
  rtl_model_path: ${BUILD_DIR}/picorv32

generation:
  min_basic_blocks: 10
  max_basic_blocks: 100
  min_block_instructions: 1
  max_block_instructions: 20
EOF

    log_info "Configuration created at $PROJECT_DIR/configs/picorv32.yaml"

    cat > "$PROJECT_DIR/configs/kronos.yaml" << EOF
# Cascade configuration for Kronos
cpu:
  name: kronos
  xlen: 32
  extensions:
    - I

memory:
  code_start: 0x00000000
  code_size: 0x10000
  data_start: 0x00010000
  data_size: 0x10000

execution:
  iss_timeout: 10000
  rtl_timeout: 100000
  spike_path: ${INSTALL_PREFIX}/bin/spike
  rtl_model_path: ${BUILD_DIR}/kronos

generation:
  min_basic_blocks: 10
  max_basic_blocks: 100
  min_block_instructions: 1
  max_block_instructions: 20
EOF

    log_info "Configuration created at $PROJECT_DIR/configs/kronos.yaml"
}

# Print PATH instructions
print_path_instructions() {
    echo ""
    log_info "Setup complete!"
    echo ""
    echo "Add the following to your shell configuration (~/.bashrc or ~/.zshrc):"
    echo ""
    echo "  export PATH=\"$INSTALL_PREFIX/bin:\$PATH\""
    echo ""
    echo "Then reload your shell or run:"
    echo ""
    echo "  source ~/.bashrc  # or ~/.zshrc"
    echo ""
    echo "To verify the installation:"
    echo ""
    echo "  spike --help"
    echo "  verilator --version"
    echo ""
    echo "To build the PicoRV32 Verilator model:"
    echo ""
    echo "  make -C $BUILD_DIR/picorv32 testbench_verilator"
    echo ""
    echo "To build the Kronos Verilator model:"
    echo ""
    echo "  make -C $BUILD_DIR/kronos verilator"
    echo ""
    echo "To run the fuzzer:"
    echo ""
    echo "  cd $PROJECT_DIR"
    echo "  source .venv/bin/activate"
    echo "  cascade --help"
    echo ""
    echo "To use the RTL model:"
    echo ""
    echo "  cascade -n 10 --cpu picorv32 --rtl-path $BUILD_DIR/picorv32"
    echo "  cascade -n 10 --cpu kronos --rtl-path $BUILD_DIR/kronos"
    echo ""
}

# Main installation
main() {
    echo "========================================"
    echo "  Cascade Dependency Setup"
    echo "========================================"
    echo ""

    case "${1:-all}" in
        all)
            check_dependencies
            install_system_deps
            install_spike
            install_verilator
            install_picorv32
            install_kronos
            install_python_deps
            create_config
            print_path_instructions
            ;;
        spike)
            check_dependencies
            install_spike
            ;;
        verilator)
            check_dependencies
            install_verilator
            ;;
        picorv32)
            install_picorv32
            ;;
        kronos)
            install_kronos
            ;;
        python)
            install_python_deps
            ;;
        *)
            echo "Usage: $0 [all|spike|verilator|picorv32|kronos|python]"
            exit 1
            ;;
    esac
}

main "$@"
