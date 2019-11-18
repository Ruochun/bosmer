from dolfin import *

def sampleMesh(res=50):
    domain_r = (
                Rectangle(dolfin.Point(0.,0.), dolfin.Point(3.*a,1.*a))
                - Rectangle(dolfin.Point(.5*a,.2*a), dolfin.Point(2.5*a,.3*a))
                - Rectangle(dolfin.Point(.5*a,.7*a), dolfin.Point(2.5*a,.8*a))
               )    
    mesh = generate_mesh(domain_r, res)
        
