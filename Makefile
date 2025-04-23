SHELL := /bin/bash
.PHONY: help config rust-version build format lint test

help:
	@echo "Makefile Commands:"
	@echo "  config               - Set up the Rust environment."
	@echo "  build                - Build chewbekka with cargo"
	@echo "  format               - Format source code with cargo fmt"
	@echo "  lint                 - Lint source code with cargo clippy"
	@echo "  test                 - Test chewbekka with cargo test"

all: format lint build test

config:
	@echo "Updating rust toolchain"
	rustup update stable
	rustup default stable

rust-version:
	@echo "Rust command-line utility versions:"
	rustc --version 			#rust compiler
	cargo --version 			#rust package manager
	rustfmt --version			#rust code formatter
	rustup --version			#rust toolchain manager
	clippy-driver --version		#rust linter

build:
	@echo "Building all projects with cargo"
	./util/build.sh

format:
	@echo "Formatting all projects with cargo"
	./util/format.sh

lint:
	@echo "Linting all projects with cargo"
	./util/lint.sh

test:
	@echo "Testing all projects with cargo"
	./util/test.sh