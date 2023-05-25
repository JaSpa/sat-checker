{
  description = "minisat tests";

  inputs.flake-utils.url = github:numtide/flake-utils;

  inputs.crane.url = github:ipetkov/crane;
  inputs.crane.inputs = {
    nixpkgs.follows = "nixpkgs";
  };

  inputs.minisat-patched-src.url = github:JaSpa/minisat;
  inputs.minisat-patched-src.flake = false;

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
    minisat-patched-src,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        overlay = final: prev: rec {
          minisat-orig = prev.minisat.overrideAttrs (o: {
            # minisat has references to zlib headers in its own headers,
            # propagation is missing in original definition.
            propagatedBuildInputs = [final.zlib];
          });

          minisat-patched = minisat-orig.overrideAttrs (o: {
            src = minisat-patched-src;
          });
        };

        inherit (pkgs) lib;

        pkgs = import nixpkgs {
          inherit system;
          overlays = [overlay];
        };

        craneLib = crane.lib.${system};

        minisat-tests = {minisat}:
          craneLib.buildPackage {
            src = craneLib.cleanCargoSource (craneLib.path ./.);

            buildInputs =
              [minisat]
              ++ lib.optionals pkgs.stdenv.isDarwin [
                # Required to avoid a linker error on darwin.
                pkgs.libiconv
              ];
          };
      in {
        formatter = pkgs.alejandra;

        packages = with pkgs; {
          inherit minisat-orig minisat-patched;
        };

        devShells.default = pkgs.mkShell {
          inputsFrom = [(minisat-tests {minisat = pkgs.minisat-orig;})];
          nativeBuildInputs = [pkgs.pkgconfig];
          packages = with pkgs; [
          ];
        };
      }
    );
}
