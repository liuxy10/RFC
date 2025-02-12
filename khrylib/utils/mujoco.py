from khrylib.utils.math import *
import xml.etree.ElementTree as ET
from pathlib import Path

from os import path

def build_body_tree(fullpath):
    if not path.exists(fullpath):
            # try the default assets path
            fullpath = path.join(Path(__file__).parent.parent, 'assets/mujoco_models', path.basename(fullpath))
            if not path.exists(fullpath):
                raise IOError("File %s does not exist" % fullpath)
    tree = ET.parse(fullpath)
    xml_string = ET.tostring(tree.getroot(), encoding='unicode')
    body_tree = {}
    try:
        root = ET.fromstring(xml_string)
        worldbody = root.find('worldbody')

        def process_body(body_element, parent_name=None):
            body_name = body_element.get('name')
            if body_name is None:
                return
            body_tree[body_name] = []
            if parent_name:
                if parent_name in body_tree:
                    body_tree[parent_name].append(body_name)
                else:
                     body_tree[parent_name] = [body_name] # Initialize if parent doesn't exist
            
            for child in body_element:
                if child.tag == 'body':
                    process_body(child, body_name)

        # Find the root body and start processing
        root_body = worldbody.find('body')
        process_body(root_body)
        
        # Ensure all parents exist in the dictionary, even if they have no children
        all_bodies = worldbody.findall('.//body')
        for body in all_bodies:
            body_name = body.get('name')
            if body_name not in body_tree:
                body_tree[body_name] = []

    except ET.ParseError as e:
        print(f"XML Parse Error: {e}")
        return {}  # Return an empty dictionary in case of an error
    except Exception as e:
        print(f"An error occurred: {e}")
        return {}

    return body_tree



def get_body_qposaddr(model):
    """ Get the qpos address of each body in the model """ 
    body_qposaddr = dict()
    for i, body_name in enumerate(model.body_names):
        start_joint = model.body_jntadr[i]
        if start_joint < 0:
            continue
        end_joint = start_joint + model.body_jntnum[i]
        start_qposaddr = model.jnt_qposadr[start_joint]
        if end_joint < len(model.jnt_qposadr):
            end_qposaddr = model.jnt_qposadr[end_joint]
        else:
            end_qposaddr = model.nq
        body_qposaddr[body_name] = (start_qposaddr, end_qposaddr)
    return body_qposaddr


def align_human_state(qpos, qvel, ref_qpos):
    qpos[:2] = ref_qpos[:2]
    hq = get_heading_q(ref_qpos[3:7])
    qpos[3:7] = quaternion_multiply(hq, qpos[3:7])
    qvel[:3] = quat_mul_vec(hq, qvel[:3])


def get_traj_pos(orig_traj):
    traj_pos = orig_traj[:, 2:].copy()
    for i in range(traj_pos.shape[0]):
        traj_pos[i, 1:5] = de_heading(traj_pos[i, 1:5])
    return traj_pos


def get_traj_vel(orig_traj, dt):
    traj_vel = []
    for i in range(orig_traj.shape[0] - 1):
        vel = get_qvel_fd(orig_traj[i, :], orig_traj[i + 1, :], dt, 'heading')
        traj_vel.append(vel)
    traj_vel.append(traj_vel[-1].copy())
    traj_vel = np.vstack(traj_vel)
    return traj_vel

if __name__ == "__main__":
    # Example usage:
    tree = ET.parse("/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/mocap_v2.xml")
    xml_string = ET.tostring(tree.getroot(), encoding='unicode')


    body_tree = build_body_tree(xml_string)

    # Print the body tree in the desired format
    print("self.body_tree = {")
    for parent, children in body_tree.items():
        print(f"    '{parent}': {children if children else []},")
    print("}")
