{
  description = "minisat tests";

  inputs.flake-utils.url = github:numtide/flake-utils;

  inputs.crane.url = github:ipetkov/crane;
  inputs.crane.inputs = {
    nixpkgs.follows = "nixpkgs";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    crane,
  }:
    flake-utils.lib.eachDefaultSystem (
      system: let
        inherit (pkgs) lib;

        pkgs = nixpkgs.legacyPackages.${system};

        craneLib = crane.lib.${system};

        minisat-tests = craneLib.buildPackage {
          src = craneLib.cleanCargoSource (craneLib.path ./.);
        };
      in {
        formatter = pkgs.alejandra;

        packages = {
          inherit minisat-tests;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
          ];
        };
      }
    );
}
