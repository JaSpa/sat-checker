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
        withDebugAttr = mkDrv: attrs @ {debugging ? false, ...}: let
          debuggingDrv = mkDrv (attrs // {debugging = true;});
        in
          if debugging
          then debuggingDrv
          else
            lib.lazyDerivation {
              derivation = mkDrv (attrs // {debugging = false;});
              passthru = {debug = debuggingDrv;};
            };

        overlay = final: prev: {
          enableDebuggingIf = cond: drv:
            if cond
            then final.enableDebugging drv
            else drv;

          minisat-orig = withDebugAttr (
            {
              debugging,
              overrides ? (_: {}),
            }: let
              drv = prev.minisat.overrideAttrs (o:
                {
                  # minisat has references to zlib headers in its own headers,
                  # propagation is missing in original definition.
                  propagatedBuildInputs = [final.zlib];
                }
                // lib.attrsets.optionalAttrs debugging {
                  cmakeFlags = (o.cmakeFlags or []) ++ ["-DCMAKE_BUILD_TYPE=Debug"];
                  dontStrip = true;
                }
                // overrides o);
            in
              final.enableDebuggingIf debugging drv
          );

          minisat-mod = withDebugAttr ({debugging}:
            final.minisat-orig {
              inherit debugging;
              overrides = o: {
                version = "${o.version}-mod";
                src = minisat-mod-src;
              };
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

        mkArgs = attrs @ {
          buildInputs ? [],
          debugging ? false,
          ...
        }:
          builtins.removeAttrs attrs ["debugging"]
          // {
            src = craneLib.cleanCargoSource (craneLib.path ./.);
            buildInputs =
              buildInputs
              ++ lib.optionals pkgs.stdenv.isDarwin [
                # Required to avoid a linker error on darwin.
                pkgs.libiconv
              ];
          }
          // lib.attrsets.optionalAttrs debugging {
            CARGO_PROFILE = "";
            dontStrip = true;
          };

        minisat-instance = withDebugAttr ({
          name,
          minisat,
          debugging,
        }: let
          cargoName = craneLib.crateNameFromCargoToml {cargoToml = ./instance/Cargo.toml;};
          attrs = mkArgs {
            inherit debugging;
            inherit (cargoName) version;
            pname = "${cargoName.pname}-${name}";

            cargoExtraArgs = "--package minisat-instance";

            buildInputs = [
              (minisat {inherit debugging;})
              pkgs.zlib
            ];
          };
        in
          # TODO: check if it is possible (or necessary?) to use
          # `enableDebugging` with crane. The problem is that the result of
          # `craneLib.buildPackage` does not have an `override` (which is used
          # by `enableDebugging`).
          pkgs.enableDebuggingIf (false && debugging) (craneLib.buildPackage attrs));

        sat-runner = withDebugAttr ({debugging}: let
          attrs =
            craneLib.crateNameFromCargoToml {cargoToml = ./runner/Cargo.toml;}
            // mkArgs {
              inherit debugging;
              cargoExtraArgs = "--package sat-runner";
            };
        in
          # TODO: check if it is possible (or necessary?) to use
          # `enableDebugging` with crane. The problem is that the result of
          # `craneLib.buildPackage` does not have an `override` (which is used
          # by `enableDebugging`).
          pkgs.enableDebuggingIf (false && debugging) (craneLib.buildPackage attrs));

        minisat-runner = withDebugAttr ({debugging}: let
          debugOrRelease = lib.getAttrFromPath (lib.optional debugging "debug");
        in
          pkgs.runCommandLocal "minisat-runner" {
            nativeBuildInputs = [pkgs.makeWrapper];
          } ''
            mkdir -p $out/bin
            makeWrapper \
                ${debugOrRelease selfp.sat-runner}/bin/sat-runner                 \
                $out/bin/minisat-runner                                           \
              --set-default MINISAT_TEST_RUNNER_INSTANCE_A                        \
                ${debugOrRelease selfp.minisat-solver.orig}/bin/minisat-instance  \
              --set-default MINISAT_TEST_RUNNER_INSTANCE_B                        \
                ${debugOrRelease selfp.minisat-solver.mod}/bin/minisat-instance
          '');

        # Combinations of all (legacy) packages defined in this flake for this system.
        selfp = (self.packages.${system} or {}) // (self.legacyPackages.${system} or {});
      in {
        formatter = pkgs.alejandra;

        legacyPackages = {
          minisat-solver = {
            orig = minisat-instance {
              name = "orig";
              minisat = selfp.minisat-orig;
            };
            mod = minisat-instance {
              name = "mod";
              minisat = selfp.minisat-mod;
            };
          };

          sat-runner = sat-runner {};

          # `minisat-runner` is a specialization of `sat-runner` to our two
          # versions of the minisat solver.
          minisat-runner = minisat-runner {};
        };

        packages = with pkgs; {
          inherit minisat-orig minisat-mod;

          link-builder = pkgs.writeShellScriptBin "build-links.sh" ''
            set -e
            prefix=result-${lib.escapeShellArg system}
            nix build .#minisat-solver.orig.debug -o "$prefix-a"
            nix build .#minisat-solver.mod.debug  -o "$prefix-b"
            nix build .#sat-runner.debug          -o "$prefix-runner"
          '';

          default = selfp.minisat-runner;
        };

        apps.default.type = "app";
        apps.default.program = "${selfp.minisat-runner}/bin/minisat-runner";

        apps.build-links.type = "app";
        apps.build-links.program = "${selfp.link-builder}/bin/build-links.sh";

        devShells.default = pkgs.mkShell {
          inputsFrom = [selfp.minisat-solver.orig selfp.sat-runner];
          packages = with pkgs;
            lib.optionals (!pkgs.stdenv.isDarwin) [
              lldb
            ];
        };
      }
    );
}
