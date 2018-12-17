
#%%
# try to simulate discrete absorption from photoreceptors

import numpy as n
import matplotlib.pyplot as p
import scipy.interpolate as i


#%%
xsurfres, ysurfres = 100,100
triangle_xsml = [0., -1./n.sqrt(2.), 1./n.sqrt(2.)]
triangle_ysml = [n.sqrt(2./3.), -n.sqrt(1./6.), -n.sqrt(1./6.)]

greenscale = 1
bluescale = 1
uvscale = 1
smooth = 3

fn = 'bombus_tuning.csv'

def triloc(gphot, bphot, uphot):
    '''Give the x, y location in the color triangle based on the receptor
    stimulation values'''
    sumstim = gphot + bphot + uphot
    if sumstim == 0: sumstim = .0001
    gphot = float(gphot) / float(sumstim)
    bphot = float(bphot) / float(sumstim)
    uphot = float(uphot) / float(sumstim)
    x1 = float(gphot - bphot)/n.sqrt(2.)
    x2 = n.sqrt(2./3.)*(uphot - (gphot + bphot)/2.)
    return (x1, x2)

def receptor_stim(spectrum, absorption, quantum=False):
    gamma = spectrum * absorption
    if quantum: gamma = n.random.poisson(gamma)
    return(sum(gamma))

def make_surf(spectrum, gr_abs, bl_abs, uv_abs, reps=10000, surfres=100):
    '''Get the whole absorption surface'''
    surf = n.zeros([surfres, surfres])
    xbounds = n.linspace(-1./n.sqrt(2.), 1./n.sqrt(2.), surfres)
    ybounds = n.linspace(-n.sqrt(1./6.), n.sqrt(2./3.), surfres)
    for i in n.arange(reps):
        gphot = receptor_stim(spectrum, gr_abs, quantum=True)
        bphot = receptor_stim(spectrum, bl_abs, quantum=True)
        uphot = receptor_stim(spectrum, uv_abs, quantum=True)
        x, y = triloc(gphot, bphot, uphot)
        surf[n.searchsorted(xbounds, x), n.searchsorted(ybounds, y)] += 1
    return xbounds, ybounds, surf


def percentile_val(surf, p):
    total = sum(surf)
    target = total - p*total
    estimate = 0
    while sum(surf[surf<estimate]) < target:
        estimate += 1
    return estimate


#%%
# wls to sample at
smooth_wls = n.arange(300., 650., 1.)
num_wls = len(smooth_wls)

# read the file with color tuning curves for each receptor
receptors = n.loadtxt(fn, skiprows=1, delimiter=',')
wl = receptors[:,0]
uv = receptors[:,1]*uvscale
bl = receptors[:,2]*bluescale
gr = receptors[:,3]*greenscale

# make smooth splines for each receptor
gr_spl = i.InterpolatedUnivariateSpline(wl, gr, k=smooth)
bl_spl = i.InterpolatedUnivariateSpline(wl, bl, k=smooth)
uv_spl = i.InterpolatedUnivariateSpline(wl, uv, k=smooth)

# and calculate the responses at our sampled wls
gr_rsp = gr_spl(smooth_wls)
bl_rsp = bl_spl(smooth_wls)
uv_rsp = uv_spl(smooth_wls)

p.figure(1, [12,6])
p.clf()
# first plot the absorption spectra
p.subplot(221)
p.plot(smooth_wls, gr_rsp, '-', color='green', lw=2)
p.plot(smooth_wls, bl_rsp, '-', color='blue', lw=2)
p.plot(smooth_wls, uv_rsp, '-', color='purple', lw=2)
p.title('Receptors')
p.xlabel(r'$\lambda$ (nm)')


# now a color
### spectral color
# spectrum = n.zeros(num_wls)
# spectrum[100] = 10
### uniform white
spectrum = n.zeros(num_wls)
spectrum[:] = .049
### white noise white
spectrum = abs(n.random.randn(num_wls))*.05

# plot it
p.subplot(223)
p.plot(smooth_wls, spectrum, '-', color='0.5', lw=2)
p.title('Light', y=.85)
p.xlabel(r'$\lambda$ (nm)')


# plot the triangle
tri = p.subplot(122, aspect='equal')
offset = .01
p.fill(triangle_xsml, triangle_ysml, 'w', lw=2, ec='k')
p.text(triangle_xsml[0], triangle_ysml[0] + offset, 'UV', horizontalalignment='center')
p.text(triangle_xsml[1], triangle_ysml[1] - offset, 'Blue', verticalalignment='top', horizontalalignment='center')
p.text(triangle_xsml[2], triangle_ysml[2] - offset, 'Green', verticalalignment='top', horizontalalignment='center')
p.xticks([])
p.yticks([])
tri.spines['left'].set_visible(False)
tri.spines['bottom'].set_visible(False)


# now get the actual color plotted on the triangle
gr_abs = receptor_stim(spectrum, gr_rsp)
bl_abs = receptor_stim(spectrum, bl_rsp)
uv_abs = receptor_stim(spectrum, uv_rsp)
x, y = triloc(gr_abs, bl_abs, uv_abs)
# and plot it
p.plot(x, y, 'ko')

# and simulate the color absorption with quantum noise to get confidence bounds
xbounds, ybounds, surf = make_surf(spectrum, gr_abs, bl_abs, uv_abs, reps=1000)

# and plot it
p.contour(xbounds, ybounds, surf.T, [percentile_val(surf, .9), percentile_val(surf, .5), percentile_val(surf, .1)])


#%%
triloc(1,2,3)


#%%
spectrum


#%%
clf()


