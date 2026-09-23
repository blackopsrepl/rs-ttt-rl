SHELL := /bin/bash
.PHONY: help all build release run test lint format clean config rust-version

help:
	@echo "Makefile commands for rs-ttt-rl:"
	@echo "  all          - Format, lint, build and test"
	@echo "  build        - Debug build with cargo"
	@echo "  release      - Optimized build with cargo (fat LTO)"
	@echo "  run          - Train GAMES games against random moves, then play"
	@echo "                 interactively (default GAMES=150000, GAMES=0 skips training)"
	@echo "  test         - Run the test suite with cargo test"
	@echo "  lint         - Lint with cargo clippy, warnings denied"
	@echo "  format       - Format sources with cargo fmt"
	@echo "  clean        - Remove build artifacts"
	@echo "  config       - Update and set the stable Rust toolchain"
	@echo "  rust-version - Print Rust toolchain versions"

all: format lint build test

build:
	cargo build

release:
	cargo build --release

GAMES ?= 150000
run:
	cargo run --release -- $(GAMES)

test:
	cargo test

lint:
	cargo clippy --all-targets --all-features -- -D warnings

format:
	cargo fmt

clean:
	cargo clean

config:
	rustup update stable
	rustup default stable

rust-version:
	@echo "Rust toolchain versions:"
	rustc --version
	cargo --version
	rustfmt --version
	rustup --version
	clippy-driver --version
