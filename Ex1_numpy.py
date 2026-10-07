import numpy as np

b_semi   = 17.5     # m
c_root   = 6.5      # m
c_tip    = 1.8      # m
CL_root  = 0.65
CL_tip   = 0.30
rho      = 0.38     # kg/m^3
V        = 240.0    # m/s
g        = 9.81     # m/s^2

# Task 1: spanwise stations and local quantities
y  = np.linspace(0.0, b_semi, 100)
c  = c_root  - (c_root  - c_tip ) * (y / b_semi) #local chord, one line, no loop
CL = CL_root - (CL_root - CL_tip) * (y / b_semi) #local lift coefficient
q  = 0.5 * rho * V**2 #dynamic pressure (a single number)
l  = 0.5 * rho * V**2 * c * CL #lift per unit span at every station

print(f"Dynamic pressure      {q:10.1f} Pa")
print(f"Root loading          {l[0]:10.1f} N/m")
print(f"Tip loading           {l[-1]:10.1f} N/m")

# Task 2: total lift and the elliptic reference
L_semi = np.trapezoid(l, y) # integrate l over y with np.trapezoid
L_full = 2.0 * L_semi
print(f"Total lift            {L_full:10.3e} N")
print(f"Implied aircraft mass {L_full / g:10.0f} kg")

shape = np.sqrt(1.0 - (y / b_semi)**2)     # unit elliptic shape
l_0   = L_semi / np.trapezoid(shape, y)   # scale factor so that the elliptic curve carries the same L_semi
l_ell = l_0 * shape

# Task 3: where is the wing working hardest
deviation = 100 * (l - l_ell) / l_ell.max()  # (l - l_ell) as a percentage of the peak elliptic loading
i_max     = np.argmax(np.abs(deviation)) # index of the largest absolute deviation (np.argmax)
print(f"Largest deviation from elliptic: {deviation[i_max]:.1f} % at y = {y[i_max]:.2f} m")

mask = CL > 0.60  # boolean mask for stations where CL(y) > 0.60
print(f"CL above 0.60 at {mask.sum()} stations, out to y = {y[mask].max():.2f} m")

# Task 4: polynomial fit of the elliptic reference and saving the results
coeffs = np.polyfit(y, l_ell, 4)  # degree-4 polynomial fit of l_ell against y
l_fit  = np.polyval(coeffs, y)    # evaluate the polynomial at the stations y
rms    = np.sqrt(np.mean((l_fit - l_ell)**2))  # root mean square difference between l_fit and l_ell
print(f"Degree-4 fit RMS error: {rms:.1f} N/m ({100 * rms / l_ell.max():.2f} % of peak)")

np.savez("spanwise_lift.npz", y=y, lift=l, lift_elliptic=l_ell)
check = np.load("spanwise_lift.npz")
print("Saved arrays:", list(check.keys()), check["lift"].shape)