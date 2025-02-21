import os
import xml.etree.ElementTree as ET

def distribute_mass(xml_file, total_mass=70.0):
    """
    Reads MuJoCo XML and distributes total mass across body segments.
    
    Args:
        xml_file: Path to MuJoCo XML file
        total_mass: Desired total mass in kg (default 70kg)
    
    Returns:
        Modified XML string with updated mass attributes
    """
   
    
    # Standard mass proportions (percentage of total mass)
    mass_ratios = {
        'root': 0.05,        # Pelvis: 5%
        'head': 0.05,        # Head: 5%
        'thorax': 0.20,      # Thorax: 20%
        'upperback': 0.15,   # Upper back: 15%
        'lowerback': 0.15,   # Lower back: 15%
        'lhumerus': 0.03,    # Left upper arm: 3%
        'rhumerus': 0.03,    # Right upper arm: 3%
        'lradius': 0.02,     # Left forearm: 2%
        'rradius': 0.02,     # Right forearm: 2%
        'lwrist': 0.01,      # Left hand: 1%
        'rwrist': 0.01,      # Right hand: 1%
        'lfemur': 0.10,      # Left thigh: 10%
        'rfemur': 0.10,      # Right thigh: 10%
        'ltibia': 0.05,      # Left lower leg: 5%
        'rtibia': 0.05,      # Right lower leg: 5%
        'lfoot': 0.02,       # Left foot: 2%
        'rfoot': 0.02        # Right foot: 2%
    }
    
    try:
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # Find all body elements
        for body in root.findall('.//body'):
            body_name = body.get('name')
            if body_name in mass_ratios:
                # Calculate mass for this body segment
                segment_mass = total_mass * mass_ratios[body_name]
                # Set mass attribute
                body.set('mass', f"{segment_mass:.3f}")
        
        return ET.tostring(root, encoding='unicode')
    
    except Exception as e:
        print(f"Error processing XML: {str(e)}")
        return None

if __name__ == '__main__':
    fdir = "/home/xliu227/Github/RFC/khrylib/assets/mujoco_models/"
    fname = "mocap_v2"
    fpath = os.path.join(fdir, fname+".xml")
    # Read and modify XML with 75kg total mass
    modified_xml = distribute_mass(fpath, total_mass=75.0)

    # Save modified XML
    if modified_xml:
        with open(os.path.join(fdir, fname+"_w_mass.xml"), 'w') as f:
            f.write(modified_xml)
