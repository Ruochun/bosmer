from dolfin import *
import generalUtilities as gU
from mshr import *
import numpy
import os

def sampleMesh(system, res=100):
    domain_r = Rectangle(Point(0.,0.), Point(32.,1.6))
    ref_num_cells = system['fluid']['recRes']
    bnd_pts = []
    for i in range(20):
        pos_x = .8*i + 8.
        pos_y = .8*((i+1)%2) + .25
        domain_r = domain_r - Rectangle(Point(pos_x, pos_y), Point(pos_x+.8, pos_y+.3))
        bnd_pts.append(pos_x+.4)
        bnd_pts.append(pos_y)
    mesh = generate_mesh(domain_r, res)
    while mesh.num_cells() < ref_num_cells:
        mesh = refine(mesh)
    return mesh, bnd_pts

def fluidBCs():

    # define your BCs
    # NOT IMPLEMENTED!
    BC['NS']['periodic'] = 'x'
    BC['NS']['essential'] = [(0, Constant(0., 0.))]

    return 0

def markSubDomains(mesh):
    subDomains = MeshFunction("size_t", mesh, mesh.topology().dim())
    subDomains.set_all(99)
    class outflowCV(SubDomain):
        def inside(self, x, on_boundary):
            return not(on_boundary) and (x[0]>28.) 
    outflowCV().mark(subDomains, 90)
    return subDomains     

def markBoundaries(mesh):
    eps = 1e-6
    boundary = MeshFunction("size_t", mesh, mesh.topology().dim()-1)
    boundary.set_all(99)
    class solidWall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary
    class inflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0]<eps
    class outflow(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and x[0]>32.-eps
    class slipWall(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and (x[1]<eps or x[1]>1.6-eps)

    solidWall().mark(boundary, 0)
    inflow().mark(boundary, 1)
    outflow().mark(boundary, 2)
    slipWall().mark(boundary, 90)
    return boundary    

def applyNSBCs(meshData, markers):
    W = meshData['fluid']['spaceNS']
    u0 = 1.0
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
        inflow = Expression(('u_in', '0.0'), u_in=u0, degree=2)
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
        inflow = Expression(('u_in', '0.0', '0.0'), u_in=u0, degree=2)
    bc0 = DirichletBC(W.sub(0), noslip, markers, 0)
    bc1 = DirichletBC(W.sub(0), inflow, markers, 1)
    bc90 = DirichletBC(W.sub(0).sub(1), 0.0, markers, 90)
    return [bc0, bc1, bc90] # slip not added

def applyAdjNSBCs(meshData, markers):
    W = meshData['fluid']['spaceNS']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc0 = DirichletBC(W.sub(0), noslip, markers, 0)
    bc1 = DirichletBC(W.sub(0), noslip, markers, 1)
    bc90 = DirichletBC(W.sub(0).sub(1), 0.0, markers, 90)
    return [bc0, bc1, bc90] # slip not added

def applyThermalBCs(meshData, markers):
    W = meshData['fluid']['spaceThermal']
    bc1 = DirichletBC(W, 0.0, markers, 1)
    return [bc1]

def applyAdjThermalBCs(meshData, markers):
    W = meshData['fluid']['spaceThermal']
    bc1 = DirichletBC(W, 0.0, markers, 1)
    return [bc1]

def applyShapeGradientBCs(meshData, markers):
    W = meshData['fluid']['spaceSG']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc1 = DirichletBC(W.sub(0), noslip, markers, 1)
    bc2 = DirichletBC(W.sub(0), noslip, markers, 2)
    bc90 = DirichletBC(W.sub(0), noslip, markers, 90)
    return [bc1, bc2, bc90]

def applyLinearElasticityBCs(meshData, markers, Var, para):
    alpha = para['stepLen']
    v = Var['fluid']['modified_v']
    W = meshData['fluid']['spaceLE']
    if W.mesh().topology().dim() == 2:
        noslip = Constant((0., 0.))
    elif W.mesh().topology().dim() == 3:
        noslip = Constant((0., 0., 0.))
    bc0 = DirichletBC(W, v, markers, 0)
    bc1 = DirichletBC(W, noslip, markers, 1)
    bc2 = DirichletBC(W, noslip, markers, 2)    
    bc90 = DirichletBC(W, noslip, markers, 90)
    return [bc0, bc1, bc2, bc90]


### remeshing utilities

def getCellNormals(mesh):
    i = 0;
    cellnormals = numpy.zeros([mesh.num_cells(),])
    for cell in cells(mesh):
        cellnormals[i] =  cell.cell_normal()[2]
        i = i+1;
    return cellnormals

def checkMeshFlip(mesh, cellnormals):
    flip=False
    i = 0
    for cell in cells(mesh):
        if cellnormals[i]*cell.cell_normal()[2] < 0.0:
            flip=True
            break;
        i = i+1;
    qmin, qmax = MeshQuality.radius_ratio_min_max(mesh)
    # return inradius and circum radius ratio. The ratio is between 0 and 1.
    return flip, qmin

# input: mesh and a point
# output: a sequence of boundary vertices that go thru the point
def getSeedVertexFromPt2D(mesh, pnt):
    tol = .0001
    boundary = BoundaryMesh(mesh,"exterior")
    coord = boundary.coordinates()
    SeedVertexId = -1
    # Find boundary vertex coincides with input pnt
    for i in edges (boundary) :
        v0 = i.entities(0)[0]
        v1 = i.entities(0)[1]
        dist_square = (coord[v0][0]-pnt[0])**2 + (coord[v0][1]-pnt[1])**2
        if dist_square < tol:
            SeedVertexId = v0;
            break
        dist_square = (coord[v1][0]-pnt[0])**2 + (coord[v1][1]-pnt[1])**2
        if dist_square < tol:
            SeedVertexId = v1;
            v1 = v0
            v0 = SeedVertexId
            break

    if SeedVertexId==-1:
        print ("cannot find boundary vertices near the point", pnt[0], pnt[1])

    return SeedVertexId


# input: mesh and a set of points with one point for each hole
# output: a set of  vertices for each  point
# NOW ONLY WORKS FOR 2D!!!
def getSeedVerticesFromPts(meshData, physics):
    mesh = meshData[physics]['mesh']
    points = meshData[physics]['bndExPts']
    hole_vertices = []
    for i in range(0,len(points)//2):
        seed = getSeedVertexFromPt2D(mesh, [points[2*i], points[2*i+1]])
        hole_vertices.append(seed)
    return hole_vertices

# input: boundary verticies for hole: one vertex for one hole
# output: verticies coordinates
# NOW ONLY WORKS FOR 2D!!!
def getSeedPtsFromVertices(meshData, physics):
    mesh = meshData[physics]['mesh']
    VID = meshData[physics]['bndVIDs']
    boundary_points = []
    for i in range(0, len(VID)):
        boundary_points.append(BoundaryMesh(mesh,"exterior").coordinates()[VID[i],0])
        boundary_points.append(BoundaryMesh(mesh,"exterior").coordinates()[VID[i],1])
    return boundary_points


# input: mesh and a point
# output: a sequence of boundary vertices that go thru the point
def getBoundaryVerticesFromPoint(mesh, pnt):
    
    tol = 0.0001**2
    #tol = 0.00005**2
    boundary = BoundaryMesh(mesh,"exterior")
    coord = boundary.coordinates()
    SeedVertexId = -1
    # Find boundary vertex coincides with input pnt
    for i in edges (boundary) :
        v0 = i.entities(0)[0]
        v1 = i.entities(0)[1]
        dist_square = (coord[v0][0]-pnt[0])**2 + (coord[v0][1]-pnt[1])**2
        if dist_square < tol:
            SeedVertexId = v0;
            break
        dist_square = (coord[v1][0]-pnt[0])**2 + (coord[v1][1]-pnt[1])**2
        if dist_square < tol:
            SeedVertexId = v1;
            v1 = v0
            v0 = SeedVertexId
            break

    BoundaryVerticies = []

    if SeedVertexId==-1:
        print ("cannot find boundary vertices near the point", pnt[0], pnt[1])
        return BoundaryVerticies

    # add 1st edge on the boundary
    BoundaryVerticies.append(v0)
    BoundaryVerticies.append(v1)


    # Find boundary vertices of a contour that contains the seed vertex
    for i in edges(boundary):
        for j in edges(boundary):
            v2 = j.entities(0)[0]
            v3 = j.entities(0)[1]
        
            if v2==v1 and v3!=v0:
                v0 = v1
                v1 = v3
                #print v0, v1
                BoundaryVerticies.append(v1)
                break;
            if v3==v1 and v2!=v0:
                v0 = v1
                v1 = v2
                #print v0, v1
                BoundaryVerticies.append(v1)
                break;
        if v1==SeedVertexId:
            print ('loop formed: number of vertices', len(BoundaryVerticies))
            return BoundaryVerticies
            break


def createMeshViaTriangle(meshData, physics):
    # number of elements and vertices may change after each re-meshing.
    # boundary_parts is re-computed after re-meshing.
    # write down boundary points, vertex ID, edge ID.
    # points[0-1] represent one point on 1st hole, points[2-3] representing a point on the second hole, and so on so forth.
    # input: mesh, points
    # output: mesh (new)
    # 1. from current mesh, idenify boundary vertices/edges, write into PSLG file
    # 2. use "triangle" to create new mesh, retain same max area.
    # 3. invoke "dolfin-convert" to convert the triangle into .xml file
    # lx, ly are only used to identify the optimization boundary (coordinates are neither lx or ly or 0)
    
    #PSLG format: https://www.cs.cmu.edu/~quake/triangle.poly.html
    #First line: <# of vertices> <dimension (must be 2)> <# of attributes> <# of boundary markers (0 or 1)>
    #Following lines: <vertex #> <x> <y> [attributes] [boundary marker]
    #One line: <# of segments> <# of boundary markers (0 or 1)>
    #Following lines: <segment #> <endpoint> <endpoint> [boundary marker]
    #One line: <# of holes>
    #Following lines: <hole #> <x> <y> (identify a point inside each hole, -xq)
    #Optional line: <# of regional attributes and/or area constraints>
    #Optional following lines: <region #> <x> <y> <attribute> <maximum area>
    mesh = meshData[physics]['mesh']
    points = meshData[physics]['bndExPts']
    ref_num_cells = meshData[physics]['initNumCells']

    scale = 1.
    maxAllowedIters = 10
    for k in range(maxAllowedIters):
        maxArea = 0
        minArea = 1000
        for i in cells(mesh):
            if i.volume() > maxArea:
                maxArea = i.volume()
            if i.volume() < minArea:
                minArea = i.volume()
        
        #scale = .5
        maxArea = maxArea*scale
        

        boundary = BoundaryMesh(mesh,"exterior")
        meshfile = open("obstaclemesh.poly", "w")
        #Writing node information for the PSLG file
        meshfile.write ( '%d \t 2 \t 0 \t 0 \n '%boundary.num_vertices())
        coor = boundary.coordinates()
        mapping = boundary.entity_map(0).array()
        for i in vertices (boundary) :
            meshfile.write('%d \t %g \t %g \n'%(i.index(), coor[i.index()][0],
                                                coor[i. index() ][1]) )

        #Writing edge information for the PSLG file
        # Planar Straight Line Graph (PSLG)
        meshfile.write('%d \t 0 \n'%boundary.num_edges())
        for i in edges(boundary):
            meshfile.write('%d \t %d \t %d \n'%(i.index(),i.entities(0)[0],i.entities(0) [1]) )

        num_holes = len(points)//2
        #Writing hole information, (cx, cy), for the PSLG f i l e
        meshfile.write('%d \n'% num_holes)
        for i in range(0,num_holes):
            boundaryVertices = getBoundaryVerticesFromPoint(mesh, [points[2*i], points[2*i+1]])
            cordx = []
            cordy = []
            for j in range(0,len(boundaryVertices)-1):
                cordx.append(coor[boundaryVertices[j]][0])
                cordy.append(coor[boundaryVertices[j]][1])
            cx = (numpy.amax(cordx)+numpy.amin(cordx))/2.
            cy = (numpy.amax(cordy)+numpy.amin(cordy))/2.
            print ("boundary points x: ", cordx)
            print ("boundary points y: ", cordy)
            #print "len cordx", len(cordx), len(cordy)
            #print "len of boundary vertices", len(boundaryVertices), cx, cy
            meshfile.write('%d \t %g \t %g\n'%(i+1, cx, cy))

        meshfile.close ()
        triangle_syntax = './triangle'
        triangle_syntax = triangle_syntax + ' -Qpqa'+str (maxArea) +' obstaclemesh'
        # Q: quiet, q: quality, c: add segments on convex hull, a: max area
        # p: PSLG file
        print ("triangle_syntax", triangle_syntax)
        os.system(triangle_syntax)
        
        # Read mesh and convert to dolfin format xml
        os.system("dolfin-convert obstaclemesh.1.node shape.xml")
        mesh = Mesh("shape.xml")
        
        print ('new mesh: num cells/elements ', mesh.num_cells(),  'num verticies ', mesh.num_vertices())
        if mesh.num_cells() > .95*ref_num_cells:
            return mesh
        else:
            scale *= .9
    info("!!!!! Maximum meshing iterations reached and still not having a sufficiently refined mesh, stopping !!!!!")
    return None

