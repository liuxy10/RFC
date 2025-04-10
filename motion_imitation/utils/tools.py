
from khrylib.utils import *
# combined with functionality of scaling model as human

import xml.etree.ElementTree as ET
import mujoco
from scipy.spatial.transform import Rotation as R

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
    
    print("lowerTrunk:", linkMass["LowerTrunk_mass"])
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


