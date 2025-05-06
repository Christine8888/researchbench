# illustrative script on how to use the GRIZY color data file. and loading TNG data. The links do not work. 

import illustris_python as il
import asdf
import numpy as np

# loading TNG catalog data
basePath = "./TNG300-1/output"
subhalos = il.groupcat.loadSubhalos(basePath, 55)
halos = il.groupcat.loadHalos(basePath, 55)

dmoPath = "./TNG300-1-Dark/output"
halos_dmo = il.groupcat.loadHalos(dmoPath, 55)
subhalos_dmo = il.groupcat.loadSubhalos(dmoPath, 55)
fp_matching = np.load("./halo_matching_055_addtree.npz")['indices_fp']
dmo_matching = np.load("./halo_matching_055_addtree.npz")['indices_dmo']

# loading the GRIZY photometry data that we generated following section 2.2
f = asdf.open("./HYDRO-TNG300-1/photo_bands_55_DESI.asdf")
borydata = f['data']
grizy = borydata['grizy']
sh_index = borydata['SubhaloIndex']
mstar = borydata['SubhaloMstar']

# generating DESI like LRGs using the colors and the TNG halo and subhalo catalogs
def desi_lrg(grizy, sh_index, mag_shift = -0.4):
    cg, cr, ci, cz, cy = grizy.T
    masks = (cr - cz > (cz + mag_shift - 16.83) * 0.45) & (cr - cz > (cz + mag_shift - 13.80) * 0.19)
    print("total number of subhalos", len(sh_index), "number of desi lrgs", len(sh_index[masks]))
    return sh_index[masks]

# getting LRG indices in the catalog
magshift = -0.4 
desi_lrg_indices_doubleslide = desi_lrg(grizy, sh_index, magshift) # these are subhalo indices in fp

# getting central and satellite indices
subhalo_grnr = subhalos['SubhaloGrNr']
_, central_inds = np.unique(subhalo_grnr, return_index=True)
lrg_indices_central_doubleslide = np.intersect1d(central_inds, desi_lrg_indices_doubleslide)
lrg_indices_satellites_doubleslide \
= desi_lrg_indices_doubleslide[~np.in1d(desi_lrg_indices_doubleslide, lrg_indices_central_doubleslide)]



# similarly to select ELG indices in the subhalo catalog we provide the following script
def desi_elg(grizy, sh_index, mag_shift = 0.0):

    # add noise 
    maglimits = np.array([24, 23.4, 22.5])
    cg = get_scatter(grizy[:,0], maglimits[0])+mag_shift
    cr = get_scatter(grizy[:,1], maglimits[1])+mag_shift
    ci = grizy[:,2]+mag_shift
    cz = get_scatter(grizy[:,3], maglimits[2])+mag_shift
    cy = grizy[:,4]+mag_shift

    masks = (cg < 23.6) & (cg > 20) & (cr - cz > 0.3) & (cr - cz < 1.6) \
            * (cg - cr < 1.15*(cr - cz) - 0.15) & (cg - cr < -1.2*(cr - cz) + 1.6) \
            * (cg < maglimits[0]) & (cr < maglimits[1]) & (cz < maglimits[2])
    print("total number of subhalos", len(sh_index), "number of desi elgs", len(sh_index[masks]))
    return sh_index[masks]

magshift = 0.6
desi_elg_indices = desi_elg(grizy, sh_index, magshift)