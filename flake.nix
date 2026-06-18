{
  description = "Python development shell with uv and ty";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = {nixpkgs, ...}: let
    systems = [
      "x86_64-linux"
      "aarch64-linux"
      "x86_64-darwin"
      "aarch64-darwin"
    ];
    forAllSystems = nixpkgs.lib.genAttrs systems;
  in {
    devShells = forAllSystems (
      system: let
        pkgs = import nixpkgs {inherit system;};
      in {
        default = pkgs.mkShell {
          packages = with pkgs; [
            uv
            ty
            black
            #
            libglvnd
            mesa
            libx11
            libxcursor
            libxext
            libxi
            libxinerama
            libxrandr
            libxrender
          ];
          shellHook = ''
            export MUJOCO_GL=''${MUJOCO_GL:-egl}
            export PYOPENGL_PLATFORM=''${PYOPENGL_PLATFORM:-egl}
            export __EGL_VENDOR_LIBRARY_FILENAMES=''${__EGL_VENDOR_LIBRARY_FILENAMES:-${pkgs.mesa}/share/glvnd/egl_vendor.d/50_mesa.json}
            export LIBGL_DRIVERS_PATH=''${LIBGL_DRIVERS_PATH:-${pkgs.mesa}/lib/dri}
            export MESA_LOADER_DRIVER_OVERRIDE=''${MESA_LOADER_DRIVER_OVERRIDE:-llvmpipe}
            export GALLIUM_DRIVER=''${GALLIUM_DRIVER:-llvmpipe}
            export MPLCONFIGDIR=''${MPLCONFIGDIR:-/tmp/matplotlib-$USER}
            export LD_LIBRARY_PATH=${pkgs.lib.makeLibraryPath [
              pkgs.libglvnd
              pkgs.mesa
              pkgs.libx11
              pkgs.libxcursor
              pkgs.libxext
              pkgs.libxi
              pkgs.libxinerama
              pkgs.libxrandr
              pkgs.libxrender
            ]}:''${LD_LIBRARY_PATH:-}
          '';
        };
      }
    );
  };
}
