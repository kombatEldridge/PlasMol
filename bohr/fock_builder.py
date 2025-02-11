def JK(wfn, D_ao):
    pot = wfn.jk.get_veff(wfn.ints_factory, 2*D_ao)
    Fa = wfn.T + wfn.Vne + pot
    return Fa

def build_fock(wfn, D_ao, exc):
    # Repisky2015.pdf Eq. 20
    ext = 0
    for dir in [0, 1, 2]:
        ext += wfn.mu[dir] * exc[dir]
    F_ao = JK(wfn, D_ao) - ext
    return F_ao