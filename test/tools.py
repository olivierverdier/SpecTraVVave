
from travwave.equations import kdv, whitham, benjamin, kawahara

# equations with monomial flux first
poly_equations = [kdv.KDV(1), kdv.KDV3(1), kdv.KDV5(1),
             whitham.Whitham(1), whitham.Whitham3(1), whitham.Whitham5(1), 
             kawahara.Kawahara(1),
             benjamin.Benjamin_Ono(1), benjamin.modified_Benjamin_Ono(1)]

equations = poly_equations[:]
equations.append(whitham.Whithamsqrt(1))
