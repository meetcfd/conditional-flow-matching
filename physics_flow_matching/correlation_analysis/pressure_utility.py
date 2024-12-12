import jax.numpy as jnp
from jax.scipy.signal import welch

class get_pressure_spectrum(object):
    """
    To calculate the wall pressure fluctuations statistics
    Input:
    data: [sample, time, nx, nz]
    Spatial and temporal resolutions

    output:
    Spectrum in different dimensions
    """



    def __init__(self,data, nx, nz, nt, Lx, Lz, dx, dz, dt):
        self.data=data
        self.nx=nx
        self.nz=nz
        self.nt=nt
        self.Lx=Lx
        self.Lz=Lz
        self.delta_kx = 2*jnp.pi/Lx
        self.delta_kz = 2*jnp.pi/Lz
        self.delta_w = 2*jnp.pi/(nt*dt)
        self.dt=dt
        self.dx=dx
        self.dz=dz

    def get_fft_all(self):
        p_wall_fluct_slice_batch_window = self.data*(jnp.hanning(self.nt)[None,:, None, None])
        pressure_fft = jnp.fft.fftn(p_wall_fluct_slice_batch_window, axes=(1, 2, 3)) * jnp.sqrt(8/3)/(self.nx*self.nz*self.nt)
        pressure_power_spectrum = jnp.abs(pressure_fft)**2 / (self.delta_kx*self.delta_kz*self.delta_w)
        return pressure_power_spectrum

    def get_fft_x(self):
        f_x = jnp.fft.fftfreq(self.nx,self.dx)[:self.nx//2]
        pressure_power_spectrum = self.get_fft_all()
        P_x = jnp.mean(jnp.sum(pressure_power_spectrum, axis=(1,3))*self.delta_kz*self.delta_w, axis=0)[:self.nx//2]
        return f_x, P_x * 2

    def get_fft_z(self):
        f_z = jnp.fft.fftfreq(self.nz,self.dz)[:self.nz//2]
        pressure_power_spectrum = self.get_fft_all()
        P_z = jnp.mean(jnp.sum(pressure_power_spectrum, axis=(1,2))*self.delta_kx*self.delta_w, axis=0)[:self.nz//2]
        return f_z, P_z * 2

    def get_fft_t(self):
        f_t = jnp.fft.fftfreq(self.nt,self.dt)[:self.nt//2]
        pressure_power_spectrum = self.get_fft_all()
        P_t = jnp.mean(jnp.sum(pressure_power_spectrum, axis=(2,3))*self.delta_kx*self.delta_kz, axis=0)[:self.nt//2]
        return f_t, P_t * 2

class get_pressure_freq_spectrum(object):
    """
    To calculate the wall pressure frequency spectrum at a given location
    Input:
    data: [nx, nz, time]

    Output:
    Spectrum at a given location
    """

    def __init__(self, data, Lx, dt, nf,rho=1, U=1):
        self.data = data
        self.nx = data.shape[0]
        self.nz = data.shape[1]
        self.nt = data.shape[2]
        self.Lx = Lx
        self.dx = self.Lx / self.nx
        self.dt = dt
        self.nf = nf
        self.rho = rho
        self.U = U
    def get_x_grid(self):
        return jnp.linspace(self.dx/2, self.Lx - self.dx/2, self.nx)

    def get_fft(self, x_loc):
        # find the index of the location
        x_grid = self.get_x_grid()
        x_idx = jnp.argmin(abs(x_grid - x_loc))

        # substract the mean value to get the fluctuations
        wallp = self.data[x_idx, :, :] - jnp.mean(self.data[x_idx, :, :],axis=-1,keepdims=True)

        # calculate the power spectrum
        f_t, P_t = welch(wallp, fs=1/self.dt, nperseg=self.nf, axis=1)
        return f_t, jnp.mean(P_t, axis=0)
    

    def get_mean_cp(self):
        mean = jnp.mean(self.data, axis=(1,2))
        cp = (mean - jnp.min(mean))/(1/2*self.rho*self.U**2)
        return cp
    
    def get_rms(self):
        wallp = self.data - jnp.mean(self.data,axis=-1,keepdims=True)
        p_rms = jnp.std(wallp, axis=(1,-1))
        return p_rms/(1/2*self.rho*self.U**2)