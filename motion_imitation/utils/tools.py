
from khrylib.utils import *
# combined with functionality of scaling model as human

import xml.etree.ElementTree as ET
import mujoco
from scipy.spatial.transform import Rotation as R
import mujoco
import numpy as np
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


def update_inertial_params(input_file, output_file, winter_params=PARAMS, total_mass=75., total_height=1.75):
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
    R_geom = R.from_quat([quat_geom[1], quat_geom[2], quat_geom[3], quat_geom[0]]).as_matrix()
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
    quat_body_inertial = R.from_matrix(R_principal).as_quat()  # [x, y, z, w]
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




def scale_torque_related_params(cfg, scale):
    print("Scaling torque related parameters by", scale)
    cfg.jkp *= scale
    cfg.jkd  *= scale
    cfg.torque_lim *= scale
    return cfg



def calculate_humanoid_height(model_path):
    # Load the model
    model = mujoco.MjModel.from_xml_path(model_path)
    # write the model in local coordinate
    data = mujoco.MjData(model)
    
    # Reset the simulation to default pose
    mujoco.mj_resetData(model, data)
    
    # reset the hip joint z direction to +- 0.3 rad
    data.qpos[6] = -0.3
    data.qpos[12] = 0.3
    # print(data.qpos)
    # Find the highest and lowest points in the model
    # First, forward kinematics to update body positions
    mujoco.mj_forward(model, data)
    
    # Initialize min and max height values
    min_height = float('inf')
    max_height = float('-inf')
    
    # Check all geoms to find the highest and lowest points
    for i in range(model.ngeom):
        # Get geom position
        geom_pos = data.geom_xpos[i]
        
        # For sphere and capsule geoms, consider their size
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_SPHERE:
            geom_height = geom_pos[2]
            geom_size = model.geom_size[i, 0]  # Radius for sphere
            min_height = min(min_height, geom_height - geom_size)
            max_height = max(max_height, geom_height + geom_size)
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            geom_height = geom_pos[2]
            geom_radius = model.geom_size[i, 0]
            min_height = min(min_height, geom_height - geom_radius)
            max_height = max(max_height, geom_height + geom_radius)
        elif model.geom_type[i] == mujoco.mjtGeom.mjGEOM_BOX:
            geom_height = geom_pos[2]
            geom_halfsize = model.geom_size[i, 2]  # z-dimension half-size
            min_height = min(min_height, geom_height - geom_halfsize)
            max_height = max(max_height, geom_height + geom_halfsize)
        else:
            # For other geom types, just use the center position
            min_height = min(min_height, geom_pos[2])
            max_height = max(max_height, geom_pos[2])
    
    # Calculate total height
    total_height = max_height - min_height
    
    return total_height

def process_body(body, linkMass, name_map):
    name = body.get("name")
    if name in name_map:
        if not isinstance(name_map[name], tuple):
            n = name_map[name]
            if n.startswith("Left"):
                n = n[4:]
            elif n.startswith("Right"):
                n = n[5:]
            mass = linkMass[n + "_mass"]
            print(f"mass of {n}:", mass)
            
        else:
            mass = 0 
            for n in name_map[name]:
                if n.startswith("Left"):
                    n = n[4:]
                elif n.startswith("Right"):
                    n = n[5:]
                mass += linkMass[n + "_mass"]
            print(f"mass of {n}:", mass)
        # ET.SubElement(body, "inertial", pos=pos, mass=mass)
        geom = body.find("geom")
        if geom is not None:
            geom.set("mass", str(mass))
        else:
            print(f"Warning: 'geom' element not found in body '{name}'")
        # print("at:", body.get("name"), "children:", [c.get("name") for c in body.findall("body")])

def assign_mass_inertia( m, input_mujoco_xml, output_mujoco_xml):
    name_map = {
        "root": "Pelvis",
        "lfemur": "LeftUpperLeg",
        "ltibia": "LeftLowerLeg",
        "lfoot": ("LeftFoot", "LeftToe"),
        "rfemur": "RightUpperLeg",
        "rtibia": "RightLowerLeg",
        "rfoot": ("RightFoot", "RightToe"),
        "upperback": "UpperTrunk",
        "thorax": "LowerTrunk",
        "lowerneck": ("Head", "Neck"),
        "lclavicle": "LeftShoulder",
        "lhumerus": "LeftUpperArm",
        "lradius": "LeftForeArm",
        "lwrist": "LeftHand",
        "rclavicle": "RightShoulder",
        "rhumerus": "RightUpperArm",
        "rradius": "RightForeArm",
        "rwrist": "RightHand"
    }
    
    linkMass = {
    "Head_mass": None,
    "Neck_mass": None,
    "UpperTrunk_mass": None,
    "LowerTrunk_mass": None,
    "Shoulder_mass": None,
    "Pelvis_mass": None,
    "UpperArm_mass": None,
    "ForeArm_mass": None,
    "Hand_mass": None,
    "UpperLeg_mass": None,
    "LowerLeg_mass": None,
    "Foot_mass": None,
    "Toe_mass": None,
    }
    linkMass = scaleMass(m, linkMass)
    
    # print("lowerTrunk:", linkMass["LowerTrunk_mass"])
    tree = ET.parse(input_mujoco_xml)
    root = tree.getroot()
    
    for body in root.findall(".//body"):
        process_body(body, linkMass, name_map)
    # print()    
    tree.write(output_mujoco_xml)

def scaleMass(TotalMass, linkMass):
    # De Leva, et. al 1996 correction
    linkMass["Head_mass"] = 0.0487 * TotalMass
    linkMass["Neck_mass"] = 0.0194 * TotalMass
    linkMass["UpperTrunk_mass"] = 0.0523 * TotalMass
    linkMass["LowerTrunk_mass"] = 0.1549 * TotalMass
    linkMass["Shoulder_mass"] = 0.0523 * TotalMass
    linkMass["Pelvis_mass"] = 0.1183 * TotalMass
    linkMass["UpperArm_mass"] = 0.0263 * TotalMass
    linkMass["ForeArm_mass"] = 0.015 * TotalMass
    linkMass["Hand_mass"] = 0.0059 * TotalMass
    linkMass["UpperLeg_mass"] = 0.1447 * TotalMass
    linkMass["LowerLeg_mass"] = 0.0457 * TotalMass
    linkMass["Foot_mass"] = (0.0133 * TotalMass) - ((0.0133 * TotalMass) / 100)
    linkMass["Toe_mass"] = 0.0133 * TotalMass / 100

    return linkMass

def scale_humanoid_model(model_path, scaled_model_path, height, weight = None):
    height_original = calculate_humanoid_height(model_path)
    scale = height / height_original
    print(f"Scaling the model size to {height} meters")
    
    # Load the model
    tree = ET.parse(model_path)
    root = tree.getroot()
    
    # Find the worldbody element
    worldbody = root.find('worldbody')
    
    def scale_entity(parent, scale):
        
        # Scale the model
        for child in parent:
            # print(child.tag)
            if child.tag == 'geom':
                # Scale the size attribute
                size = child.attrib['size']
                size = [float(x) for x in size.split(' ')]
                size = [x * scale for x in size]
                child.set('size', ' '.join(str(x) for x in size))
                # fromto
                fromto = child.attrib.get('fromto')
                if fromto:
                    fromto = [float(x) for x in fromto.split(' ')]
                    fromto = [x * scale for x in fromto]
                    child.set('fromto', ' '.join(str(x) for x in fromto))
                pos = child.attrib.get('pos')
                if pos:
                    print(pos)
                    pos = [float(x) for x in pos.split(' ')]
                    pos = [x * scale for x in pos]
                    child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
            elif child.tag == 'body':
                # Scale the pos attribute
                # print(child.attrib['name'])
                try:
                    pos = child.attrib['pos']
                except KeyError:
                    pos = "0 0 0"
                pos = [float(x) for x in pos.split(' ')]
                pos = [x * scale for x in pos]
                child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
            elif child.tag == 'joint':
                # Scale the pos attribute
                try:
                    pos = child.attrib['pos']
                except KeyError:
                    pos = "0 0 0"
                pos = [float(x) for x in pos.split(' ')]
                pos = [x * scale for x in pos]
                child.set('pos', ' '.join(str(x) for x in pos))
                scale_entity(child, scale)
                
    # Scale the model
    scale_entity(worldbody, scale)       
    
    # reset the total mass
    compiler = tree.find('compiler')
    # print(f"compiler: {compiler.attrib}")
    if weight is not None:
        if type(weight) is float or type(weight) is int:
            compiler.set('settotalmass', str(weight)) # set the total mass
            print(f"total mass: {weight} kg")
        elif type(weight) is dict:
            assign_inertial_to_bodies(worldbody, weight)
                
        # print(f"total mass: {weight} kg")
    # compiler.attrib.pop('autolimits') #remove the autolimits attribute
    
    # Save the scaled model
    
    print(f"Saving the scaled model to {scaled_model_path}")
    tree.write(scaled_model_path)

def assign_inertial_to_bodies(parent, mass_dict):
    # Scale the model
    for child in parent:
        if child.tag == 'body':
            # Scale the pos attribute
            body_name = child.attrib['name']
            if body_name in mass_dict:
                mass = mass_dict[body_name]
                # child.set('mass', str(mass))
                mass_element = child.find('mass')
                if mass_element is None:
                    mass_element = ET.SubElement(child, 'mass')
                    mass.set('value', str(mass))
            assign_inertial_to_bodies(child, mass_dict)
        else:
            pass
    
def print_body_info_table(model):
    sum_mass = np.sum(model.body_mass)
    print("Body Information")
    print("Body Name          | Body Mass ")
    print("-------------------|-------------------------")
    for i in range(model.nbody):
        # print(f"{model.body_id2name(i):<19}{model.body_mass[i]:<19}")
        print(f"Body {i}: {model.body(i).name:<19} | {model.body_mass[i]:<19}")
    print(f"Total Mass: {sum_mass:.4f}")

def remove_autolimits_attribute(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    compiler = root.find('compiler')
    if 'autolimits' in compiler.attrib:
        compiler.attrib.pop('autolimits')
    tree.write(xml_file)



def get_expert(expert_qpos, expert_meta, env):
    old_state = env.sim.get_state()
    expert = {'qpos': expert_qpos, 'meta': expert_meta}
    feat_keys = {'qvel', 'rlinv', 'rlinv_local', 'rangv', 'rq_rmh',
                 'com', 'head_pos', 'ee_pos', 'ee_wpos', 'bquat', 'bangvel'}
    for key in feat_keys:
        expert[key] = []

    for i in range(expert_qpos.shape[0]):
        qpos = expert_qpos[i]
        env.data.qpos[:] = qpos
        env.sim.forward()
        rq_rmh = de_heading(qpos[3:7])
        ee_pos = env.get_ee_pos(env.cfg.obs_coord) # ee_pos in obs_coord
        ee_wpos = env.get_ee_pos(None)
        bquat = env.get_body_quat()
        com = env.get_com()
        head_pos = env.get_body_frame_position('head').copy()
        if i > 0:
            prev_qpos = expert_qpos[i - 1]
            qvel = get_qvel_fd_new(prev_qpos, qpos, env.dt) # finite difference
            qvel = qvel.clip(-10.0, 10.0)
            rlinv = qvel[:3].copy()
            rlinv_local = transform_vec(qvel[:3].copy(), qpos[3:7], env.cfg.obs_coord)
            rangv = qvel[3:6].copy()
            expert['qvel'].append(qvel) 
            expert['rlinv'].append(rlinv) 
            expert['rlinv_local'].append(rlinv_local)
            expert['rangv'].append(rangv)
        expert['ee_pos'].append(ee_pos) 
        expert['ee_wpos'].append(ee_wpos)
        expert['bquat'].append(bquat)
        expert['com'].append(com)
        expert['head_pos'].append(head_pos)
        expert['rq_rmh'].append(rq_rmh)
    expert['qvel'].insert(0, expert['qvel'][0].copy())
    expert['rlinv'].insert(0, expert['rlinv'][0].copy())
    expert['rlinv_local'].insert(0, expert['rlinv_local'][0].copy())
    expert['rangv'].insert(0, expert['rangv'][0].copy())
    # get expert body quaternions
    for i in range(1, expert_qpos.shape[0]):
        bangvel = get_angvel_fd(expert['bquat'][i - 1], expert['bquat'][i], env.dt)
        expert['bangvel'].append(bangvel)
    expert['bangvel'].insert(0, expert['bangvel'][0].copy())

    for key in feat_keys:
        expert[key] = np.vstack(expert[key])
    expert['len'] = expert['qpos'].shape[0]
    expert['height_lb'] = expert['qpos'][:, 2].min()
    expert['head_height_lb'] = expert['head_pos'][:, 2].min()
    if expert_meta['cyclic']:
        expert['init_heading'] = get_heading_q(expert_qpos[0, 3:7])
        expert['init_pos'] = expert_qpos[0, :3].copy()
    env.sim.set_state(old_state)
    env.sim.forward()
    return expert

if __name__ == "__main__":
    # update_inertial_params('khrylib/assets/mujoco_models/mocap_v2_local.xml',
    #                         'khrylib/assets/mujoco_models/mocap_v2_local_new.xml',  winter_params = PARAMS, total_mass = 75., total_height=1.75)
    modify_xml_local_coordinate('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v3_all_scaled.xml',
                                '/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v3_all_scaled_local.xml')
    
    update_inertial_params('/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v3_all_scaled_local.xml',
                            '/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v3_all_scaled_local.xml',  winter_params = PARAMS, total_mass = 75., total_height=1.75)



