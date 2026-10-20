if (process.platform === 'darwin') await import('./build-accelerate.ts');
else if (process.platform === 'linux') await import('./build-openblas.ts');
else throw new Error('Native acceleration supports macOS (Accelerate) and Linux (OpenBLAS).');
export {};
