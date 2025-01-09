def JK(wfn, D):
    pot = wfn.jk.get_veff(wfn.ints_factory, 2*D)
    Fa = wfn.T + wfn.Vne + pot
    return Fa
