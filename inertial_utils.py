

import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from khrylib.utils.transformation import quaternion_from_matrix, rotation_from_quaternion
PARAMS = {
    # COM: center of mass /segment length in geom frame. MASS: segment mass / total mass. I: inertia: MP COG/seg len, AP COG/seg len, TWIST COG/seg len
    'root': {'com': [0., 0., 0.],'mass': 0.142, 'I': [-1., -1., -1.]},
    'lfemur': {'com': [0., 0., 0.5 - 0.433],'mass': 0.100, 'I': [0.323, 0.323, -1]},
    'ltibia': {'com': [0., 0., 0.5 - 0.433],'mass': 0.0465, 'I': [0.302, 0.302, -1]},
    'lfoot': {'com': [0., 0., 0.],'mass': 0.0145, 'I': [0.475, -1., 0.475]},
    'rfemur': {'com': [0., 0., 0.5 - 0.433],'mass': 0.100, 'I': [0.323, 0.323, -1]},
    'rtibia': {'com': [0., 0., 0.5 - 0.433],'mass': 0.0465, 'I': [0.302, 0.302, -1]},
    'rfoot': {'com': [0., -0., 0.],'mass': 0.0145, 'I': [0.475, -1., 0.475]},
    'upperback': {'com': [0., 0., 0.],'mass': 0.139, 'I': [-1., -1., -1.]},
    'thorax': {'com': [0., 0., 0.],'mass': 0.216, 'I': [-1., -1., -1.]},
    'lowerneck': {'com': [0., 0., 0.],'mass': 0.081, 'I': [-1., -1., -1.]},
    "lclavicle": {'com': [0., 0., 0.],'mass': 0.001, 'I': [-1., -1., -1.]},
    "rclavicle": {'com': [0., 0., 0.],'mass': 0.001, 'I': [-1., -1., -1.]},
    "lhumerus": {'com': [0., 0., 0.5 - 0.436],'mass': 0.028, 'I': [0.322, 0.322, -1.]},
    "rhumerus": {'com': [0., 0., 0.5 - 0.436],'mass': 0.028, 'I': [0.322, 0.322, -1.]},
    "lradius": {'com': [0., 0., 0.5 - 0.43],'mass': 0.016, 'I': [0.303, 0.303, -1.]},
    "rradius": {'com': [0., 0., 0.5 - 0.43],'mass': 0.016, 'I': [0.303, 0.303, -1.]},
    "lwrist": {'com': [0., 0., 0.],'mass': 0.006, 'I': [-1., -1., -1.]},
    "rwrist": {'com': [0., 0., 0.],'mass': 0.006, 'I': [-1., -1., -1.]}
}

# print("total mass", np.sum([v['mass'] for v in PARAMS.values()]))

def calculate_inertia(rho, m, R=None):

    rho = np.array(rho)
    mask = rho>0  
    I = np.zeros(3)
    if np.sum(mask) == 3: # 001, 011, 111
        I[mask] = rho[mask]**2 * m
    elif np.sum(mask) == 2:
        assert R is not None, "R should be provided when two inertia values are non-zero"
        I[mask] = rho[mask]**2 * m
        I[~mask] = max(1/2 * m * R**2, 1/2* I[mask][0]) # R is the radius of the cylinder
    elif np.sum(mask) == 1:
        raise NotImplementedError("TODO: implement this case")
    else:
        assert np.sum(mask) == 0, "All dimensions should be zero"
        density = 1.0* 1e3 # kg/m^3
        r = np.cbrt((3 * m) / (4 * np.pi * density))
        print("estimated radius", r)
        I[~mask] = (2/5) * m * r**2
    return I


def update_inertial_params(input_file, output_file, winter_params, total_mass=75., total_height=1.75):
    # Load and modify XML
    tree = ET.parse(input_file)
    root = tree.getroot()
    # Disable auto inertia computation
    compiler = root.find('compiler')
    compiler.set('inertiafromgeom', 'false')
    
    # Set global coordinate system alignment
    compiler.set('coordinate', 'local')
    worldbody = tree.getroot().find('worldbody')
    # get the root body
    # root_body = root.find('body[@name="root"]')
    for body in worldbody.findall('.//body'):
        name = body.get('name')
        print(f"Processing body: {name}")
         # Remove existing inertial if present
        inertial = body.find('inertial')
        if inertial is not None:
            body.remove(inertial)
        if name in winter_params and body.find('geom') is not None:
            params = winter_params[name] 
            geom = body.find('./geom')
            # delete mass in geom
            if geom is not None and geom.get('mass') is not None:
                del geom.attrib['mass']  
        else:
            print(f"Warning: {name} not in winter_params, skipping.")
            continue
        # transform inertial frame from geom to body
        geom = body.find('./geom')
        if geom is not None:
            quat_geom = np.array([float(x) for x in geom.get('quat',"1. 0. 0. 0.").split()])
            # the geom start and end around com
            pos_geom = np.array([float(x) for x in geom.get('pos',"0. 0. 0.").split()])
            if geom.get('type') == 'capsule':
                s = np.array([float(x) for x in geom.get('size',"0.1 0.1").split()])
                if s.shape[0] == 2:
                    seg_len = s[1]
                    cylinder_radius = np.array([float(x) for x in geom.get('size',"0.1 0.1").split()])[0]       
                elif s.shape[0] ==1:
                    cylinder_radius  = s[0]
                    fromto = np.array([float(x) for x in geom.get('fromto',"0. 0. 0. 0. 0. 0.").split()])
                    seg_len = np.linalg.norm(fromto[3:] - fromto[:3])
            elif geom.get('type') == 'box':
                s = np.array([float(x) for x in geom.get('size',"0.1 0.1 0.1").split()]) 
                seg_len = s[1]  # for box, the length is the second dimension
                cylinder_radius = np.sqrt((s[0]**2 + s[2])/6) # equivalent radius of the box
            else:
                seg_len = np.array([float(x) for x in geom.get('size',"0.1").split()])[0] 
                cylinder_radius = None #np.sqrt(np.sum(size[[0,2]]**2)/6) # equivalent radius of the box
            seg_mass = params['mass'] * total_mass
            params['com'] = np.array(params['com'])  # Ensure 'com' is a NumPy array
            # print dimension info
            print("seg_len", seg_len, "cylinder_radius", cylinder_radius, "seg_mass", seg_mass)
            I = calculate_inertia(np.array(params['I']) * seg_len, seg_mass, R = cylinder_radius)
            # I = np.array(params['I']) * seg_len
            print("I", I)
            com_body, eigvals, quat= transform_inertial_geom_to_body(
                seg_mass, params['com']* seg_len , I, quat_geom, pos_geom )
            # print(quat_geom, quat)
            # Create new inertial element
            inertial = ET.SubElement(body, 'inertial')
            inertial.set('pos', ' '.join(map(str, com_body)))
            inertial.set('mass', str(seg_mass))
            inertial.set('diaginertia', ' '.join(map(str, eigvals)))
            inertial.set('quat', ' '.join(map(str, quat)))
            
                
    # Save modified model
    tree.write(output_file, encoding='utf-8', xml_declaration=True)
    print(f"Updated inertial parameters and save at:{output_file}")



def transform_inertial_geom_to_body(
    mass, com_geom, diaginertia_geom, quat_geom, pos_geom
):
    """
    Transform inertial properties from geom frame to body frame.
    Args:
        mass: float
        com_geom: (3,) array, center of mass in geom frame
        diaginertia_geom: (3,) array, diagonal inertia in geom frame
        quat_geom: (4,) array, [w, x, y, z] quaternion of geom frame in parent (body) frame
        pos_geom: (3,) array, position of geom frame in body frame
        quat_body: (4,) array, [w, x, y, z] quaternion of body frame in parent frame (usually identity)
        pos_body: (3,) array, position of body frame in parent frame (usually zero)
    Returns:
        com_body: (3,) array, center of mass in body frame
        diaginertia_body: (3,) array, diagonal inertia in body frame
        quat_body_inertial: (4,) array, orientation of principal axes in body frame
    """
    # Rotation from geom to body
    R_geom = Rotation.from_quat([quat_geom[1], quat_geom[2], quat_geom[3], quat_geom[0]]).as_matrix()
    # R_geom = rotation_from_quaternion(quat_geom)
    # Inertia tensor in geom frame
    I_geom = np.diag(diaginertia_geom)
    # Rotate inertia tensor to body frame
    I_body = R_geom @ I_geom @ R_geom.T
    # Center of mass in body frame
    com_body = R_geom @ com_geom + pos_geom
    # Parallel axis theorem (if com_geom is not at origin)
    d = com_body - pos_geom  # vector from geom origin to COM in body frame
    I_body += mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))
    # Get principal axes and diagonalize
    eigvals, eigvecs = np.linalg.eigh(I_body)
    # Sort by descending eigenvalue
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Quaternion for principal axes
    R_principal = eigvecs
    assert np.allclose(R_principal.T @ R_principal, np.eye(3)), "Axes are not orthogonal"
    assert np.allclose(abs(np.linalg.det(R_principal)), 1), "Axes are not normalized"
    if np.linalg.det(R_principal) < 0:
        R_principal[:, [1, 2]] = R_principal[:, [2, 1]]  # flip the order of 2nd and 3rd columns  assert np.linalg.det(R_principal) > 0, "Axes are not right-handed"
        eigvals [[1,2]] = eigvals [[2,1]]
    quat_body_inertial = Rotation.from_matrix(R_principal).as_quat()  # [x, y, z, w]
    # quat_body_inertial = quaternion_from_matrix(R_principal)
    # Convert to MuJoCo [w, x, y, z]
    quat_body_inertial = [quat_body_inertial[3], quat_body_inertial[0], quat_body_inertial[1], quat_body_inertial[2]]
    return com_body, eigvals, quat_body_inertial 




def modify_xml_local_coordinate(input_file, output_file): 
    """
    This script provides a function to modify an XML file based on specific requirements. 
    The modifications include adding, deleting, and updating elements and attributes in the XML structure.
    Functions:
        modify_xml(input_file: str, output_file: str) -> None:
            Modifies the input XML file according to the following rules:
            1. Adds an <include> tag with the attribute {"file": "./common/materials.xml"} 
            at the beginning of the <mujoco> field.
            2. Deletes the attribute with the name "autolimits" in the <compiler> field, if it exists.
            3. Deletes the entire <asset> field, including all its subfields.
            4. Deletes the attribute with the name "gravcomp" in all fields, if it exists.
            5. Deletes the attribute with the name "actdim" in the <general> field, if it exists.
    Parameters:
        input_file (str): The path to the input XML file to be modified.
        output_file (str): The path to save the modified XML file.
    Returns:
        None: The function modifies the XML file in place and writes the changes to the output file.
    Usage:
        Call the `modify_xml` function with the input and output file paths to apply the modifications.
    """
    model = mujoco.MjModel.from_xml_path(input_file)
    mujoco.mj_saveLastXML(output_file, model)
    # Parse the XML file
    tree = ET.parse(output_file)
    root = tree.getroot()

    # 1. Add <include> tag in <mujoco> field
    include_tag = ET.Element("include", {"file": "./common/materials.xml"})
    root.insert(0, include_tag)

    # 2. Delete attribute with name "autolimits" in <compiler> field
    compiler = root.find("compiler")
    if compiler is not None and "autolimits" in compiler.attrib:
        del compiler.attrib["autolimits"]
    # 3. Delete all subfields in <asset> field, including <asset /> itself
    asset = root.find("asset")
    if asset is not None:
    #     for child in list(asset):
    #         asset.remove(child)
        root.remove(asset)

    # 4. In all fields, delete attribute with name "gravcomp"
    for elem in root.iter():
        if "gravcomp" in elem.attrib:
            del elem.attrib["gravcomp"]

    # 5. In <general> field, delete attribute with name "actdim"
    for general in root.findall(".//general"):
        if "actdim" in general.attrib:
            del general.attrib["actdim"]

    # Write the modified XML to the output file
    tree.write(output_file, encoding="utf-8", xml_declaration=True) 
    print(f"Modified XML file saved at: {output_file}")
if __name__ == "__main__":
    winter_params = {
    'root':{
        'com': [0, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'thorax': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lfemur': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'ltibia': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lfoot': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rfemur': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rtibia': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rfoot': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lhumerus': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lradius': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lwrist': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rhumerus': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rradius': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'rwrist': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'lowerneck':{
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    },
    'head': {
        'com': [0.1, 0, 0],
        'mass': 1.0,
        'I': [0.1, 0.1, 0.1]
    }
    }
    # update_inertial_params('khrylib/assets/mujoco_models/mocap_v2_local.xml',
    #                         'khrylib/assets/mujoco_models/mocap_v2_local_new.xml',  winter_params = PARAMS, total_mass = 75., total_height=1.75)
    modify_xml_local_coordinate('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v2_all_scaled.xml',
                                '/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v2_all_scaled_local.xml')
    
    update_inertial_params('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v2_all_scaled_local.xml',
                            '/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v2_all_scaled_local.xml',  winter_params = PARAMS, total_mass = 75., total_height=1.75)

    # # Example usage:
    # mass = 1.0
    # com_geom = np.array([0.1, 0, 0])  # center of mass in geom frame
    # diaginertia_geom = np.array([0.1, 0.1, 0.1])
    # quat_geom = [1, 0, 0, 0]  # identity quaternion [w, x, y, z]
    # pos_geom = np.array([0.2, 0.3, 0.4])  # geom position in body frame
    # quat_body = [1, 0, 0, 0]  # usually identity
    # pos_body = np.array([0, 0, 0])

    # com_body, diaginertia_body, quat_body_inertial = transform_inertial_geom_to_body(
    #     mass, com_geom, diaginertia_geom, quat_geom, pos_geom)
    
    # print("COM in body frame:", com_body)
    # print("Diagonal inertia in body frame:", diaginertia_body)
    # print("Inertial orientation in body frame (quat):", quat_body_inertial)
