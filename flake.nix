{
  description = "minisat tests";

  inputs.flake-utils.url = github:numtide/flake-utils;

  inputs.rust-overlay.url = github:oxalica/rust-overlay;
  inputs.rust-overlay.inputs = {
    nixpkgs.follows = "nixpkgs";
  };

  inputs.crane.url = github:ipetkov/crane;
  inputs.crane.inputs = {
    nixpkgs.follows = "nixpkgs";
  };

  inputs.minisat-mod-src.url = github:JaSpa/minisat;
  inputs.minisat-mod-src.flake = false;

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
    rust-overlay,
    minisat-mod-src,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlay = final: prev: rec {
          minisat-orig = final.enableDebugging (prev.minisat.overrideAttrs (o: {
            # minisat has references to zlib headers in its own headers,
            # propagation is missing in original definition.
            propagatedBuildInputs = [final.zlib];
            cmakeFlags = (o.cmakeFlags or []) ++ ["-DCMAKE_BUILD_TYPE=Debug"];
            dontStrip = true;
          }));

          minisat-mod = minisat-orig.overrideAttrs (o: {
            version = "${o.version}-mod";
            src = minisat-mod-src;
          });
        };

        inherit (pkgs) lib;

        pkgs = import nixpkgs {
          inherit system;
          overlays = [
            overlay
            (import rust-overlay)
          ];
        };

        craneLib = (crane.mkLib pkgs).overrideToolchain pkgs.rust-bin.nightly."2023-05-24".default;

        mkArgs = attrs @ {buildInputs ? [], ...}:
          attrs
          // {
            src = craneLib.cleanCargoSource (craneLib.path ./.);
            buildInputs =
              buildInputs
              ++ lib.optionals pkgs.stdenv.isDarwin [
                # Required to avoid a linker error on darwin.
                pkgs.libiconv
              ];
          };

        minisat-instance = {
          name,
          minisat,
        }: let
          cargoName = craneLib.crateNameFromCargoToml {cargoToml = ./instance/Cargo.toml;};
          attrs = mkArgs {
            inherit (cargoName) version;
            pname = "${cargoName.pname}-${name}";

            cargoExtraArgs = "--workspace --exclude minisat-test-runner";
            CARGO_PROFILE = "";
            dontStrip = true;

            buildInputs = [
              minisat
              pkgs.zlib
            ];
          };
        in
          craneLib.buildPackage attrs;

        minisat-test-runner = let
          attrs =
            mkArgs {cargoExtraArgs = "--workspace --exclude minisat-instance";}
            // craneLib.crateNameFromCargoToml {cargoToml = ./runner/Cargo.toml;};
        in
          craneLib.buildPackage attrs;

        selfp = self.packages.${system};
      in {
        formatter = pkgs.alejandra;

        packages = with pkgs; {
          inherit minisat-orig minisat-mod minisat-test-runner;

          minisat-instance-orig = minisat-instance {
            name = "orig";
            minisat = selfp.minisat-orig;
          };

          minisat-instance-mod = minisat-instance {
            name = "mod";
            minisat = selfp.minisat-mod;
          };

          minisat-runner =
            pkgs.runCommandLocal "minisat-runner" {
              nativeBuildInputs = [pkgs.makeWrapper];
            } ''
              mkdir -p $out/bin
              makeWrapper \
                  ${selfp.minisat-test-runner}/bin/minisat-test-runner        \
                  $out/bin/minisat-runner                                     \
                --set-default MINISAT_TEST_RUNNER_INSTANCE_A                  \
                  ${selfp.minisat-instance-orig}/bin/minisat-instance         \
                --set-default MINISAT_TEST_RUNNER_INSTANCE_B                  \
                  ${selfp.minisat-instance-mod}/bin/minisat-instance
            '';
        };

        apps.default.type = "app";
        apps.default.program = "${selfp.minisat-runner}/bin/minisat-runner";

        devShells.default = pkgs.mkShell {
          inputsFrom = [selfp.minisat-instance-orig minisat-test-runner];
          packages = with pkgs;
            lib.optionals (!pkgs.stdenv.isDarwin) [
              lldb
            ];
        };
      }
    );
}
