import os
import subprocess
import re
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext

class CUDAExtension(Extension):
    def __init__(self, name, sources, output_name=None, **kwargs):
        # We pass an empty list for sources to the parent to avoid standard compilation
        super().__init__(name, sources=[], **kwargs)
        self.cuda_sources = sources
        self.output_name = output_name

class build_cuda_ext(build_ext):
    def run(self):
        # Check if nvcc is available
        try:
            subprocess.check_output(['nvcc', '--version'])
            has_nvcc = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            has_nvcc = False
            print("WARNING: nvcc not found. Skipping CUDA compilation. GPU features will not be available.")

        if has_nvcc:
            for ext in self.extensions:
                if isinstance(ext, CUDAExtension):
                    self.build_cuda_extension(ext)
        
        # Filter out CUDAExtension before running standard build_ext
        # so it doesn't complain about empty sources
        self.extensions = [ext for ext in self.extensions if not isinstance(ext, CUDAExtension)]
        
        if self.extensions:
            super().run()

    def get_cuda_version(self):
        try:
            version_out = subprocess.check_output(['nvcc', '--version']).decode()
            match = re.search(r'release (\d+)\.', version_out)
            if match:
                return int(match.group(1))
        except Exception:
            pass
        return 11 # Default fallback

    def get_host_gpu_arch(self):
        """Try to detect the host GPU architecture using nvidia-smi."""
        try:
            # Try to get compute capability (e.g., 7.0)
            # nvidia-smi --query-gpu=compute_cap --format=csv,noheader
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=compute_cap', '--format=csv,noheader']).decode()
            match = re.search(r'(\d+)\.(\d+)', out)
            if match:
                return match.group(1) + match.group(2)
        except Exception:
            pass
        return None

    def build_cuda_extension(self, ext):
        cuda_version = self.get_cuda_version()
        
        # Determine target architectures
        target_archs = set()

        # 1. Allow user override via environment variable
        env_arch = os.environ.get('PICASO_CUDA_ARCH')
        if env_arch:
            target_archs.update(env_arch.split(','))
        else:
            # 2. Try to detect local host architecture
            host_arch = self.get_host_gpu_arch()
            if host_arch:
                target_archs.add(host_arch)
            
            # 3. Add sensible defaults based on CUDA version
            # Architecture 35 (Kepler) was deprecated/removed in newer CUDA
            if cuda_version < 12:
                target_archs.add('35')
            
            target_archs.update(['50', '60', '70', '75'])
            
            if cuda_version >= 11:
                target_archs.update(['80', '86'])
            if cuda_version >= 12:
                target_archs.add('90')

        # Filter out architectures that might not be supported by the current nvcc
        # (Very old architectures removed in very new CUDA, or very new archs on old CUDA)
        final_archs = sorted(list(target_archs))

        for source in ext.cuda_sources:
            # Determine output shared library name
            if ext.output_name:
                output = os.path.join(os.path.dirname(source), ext.output_name)
            else:
                base = os.path.splitext(source)[0]
                output = f"{base}.so"
            
            # Build nvcc command
            cmd = ['nvcc', '-o', output, '-shared', '-Xcompiler', '-fPIC']
            
            # Add gencode flags
            for arch in final_archs:
                # Basic check to avoid known incompatibilities
                if cuda_version >= 12 and int(arch) < 50:
                    continue # Skip Kepler and older on CUDA 12+
                
                cmd.extend(['-gencode', f'arch=compute_{arch},code=sm_{arch}'])
            
            # Add JIT compatibility for future/higher archs using the highest supported compute arch
            highest_arch = final_archs[-1]
            cmd.extend(['-gencode', f'arch=compute_{highest_arch},code=compute_{highest_arch}'])
            
            cmd.append(source)
            
            print(f"Compiling CUDA extension: {' '.join(cmd)}")
            try:
                subprocess.check_call(cmd)
            except subprocess.CalledProcessError as e:
                print(f"ERROR: Compilation failed for {source}: {e}")
                # If it failed due to unsupported architecture, we might try to fall back or just report
                raise

setup(
    ext_modules=[
        CUDAExtension('picaso.gpu.thermal_1d_at_top', 
                      ['picaso/gpu/thermal_1d_kernel_at_top.cu'], 
                      output_name='thermal_1d_kernel_at_top.so'),
        CUDAExtension('picaso.gpu.thermal_1d_ver1', 
                      ['picaso/gpu/thermal_1d_kernel.cu'], 
                      output_name='thermal_1d_kernel_ver1.so'),
        CUDAExtension('picaso.gpu.reflect_1d_ver1', 
                      ['picaso/gpu/reflect_1d_kernel.cu'], 
                      output_name='reflect_1d_kernel_ver1.so'),
        CUDAExtension('picaso.gpu.transit_1d', 
                      ['picaso/gpu/transit_1d_kernel.cu'], 
                      output_name='transit_1d_kernel.so'),
    ],
    cmdclass={
        'build_ext': build_cuda_ext,
    }
)

