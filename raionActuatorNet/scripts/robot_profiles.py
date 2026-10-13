"""Explicit stand-27 variants of existing robot scenes; never edit source XMLs."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
MODELS=ROOT.parent/'model'
CATALOG={
    'test': ROOT/'tests/pendulum.xml',
    'rough-bipedal': MODELS/'RoughTerrain-3/RoughTerrain-3_bipedal.xml',
    'rough-quadrupedal': MODELS/'RoughTerrain-3/RoughTerrain-3_quadrupedal.xml',
}


def prepare(name):
    source=CATALOG[name]
    if name=='test': return source
    digest=hashlib.sha256(source.read_bytes()).hexdigest()
    folder=ROOT/'models'/name
    output=folder/('stand27_'+digest[:12]+'.xml')
    metadata=output.with_suffix('.json')
    if output.exists():
        saved=json.loads(metadata.read_text())
        if saved['output_sha256'] != hashlib.sha256(output.read_bytes()).hexdigest():
            raise ValueError('prepared model changed; do not reuse stale metadata')
        return output
    root=ET.parse(source).getroot()
    compiler=root.find('compiler')
    for attribute in ['meshdir','texturedir']:
        compiler.set(attribute,str((source.parent/compiler.get(attribute,'.')).resolve()))
    # Preserve the source nominal pose, dropping only root/free and fixed gripper coordinates.
    joints=root.findall('.//worldbody//joint')
    drops={'L_gripper_J','R_gripper_J'}
    names=[j.get('name') for j in joints]
    free=root.findall('.//freejoint')
    if len(free)!=1 or len(names)!=29 or not drops.issubset(names):
        raise ValueError('unexpected robot topology; review before preparing stand variant')
    for parent in root.iter():
        for child in list(parent):
            if child.tag=='freejoint' or (child.tag=='joint' and child.get('name') in drops):
                parent.remove(child)
    for tag in ['equality','actuator','sensor']:
        for child in root.findall(tag): root.remove(child)
    for key in root.findall('./keyframe/key'):
        q=key.get('qpos','').split()
        if len(q)!=7+len(names): raise ValueError('unexpected keyframe layout')
        values=[q[7+i] for i,n in enumerate(names) if n not in drops]
        key.attrib.clear()
        key.set('name','stand_nominal');key.set('qpos',' '.join(values))
    root.set('model',name+'_STAND_27_NO_CONTACT_NO_GRIPPER')
    folder.mkdir(parents=True,exist_ok=True)
    ET.indent(root)
    ET.ElementTree(root).write(output,encoding='utf-8',xml_declaration=True)
    info=dict(source=str(source),source_sha256=digest,
              output_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
              changes=['root freejoint removed: root body world pose retained',
                       'two gripper slide joints fixed at their zero position, bodies/masses retained',
                       'equality removed, grippers are now rigid',
                       'native actuators and sensors removed: external joint torques used',
                       'nominal keyframe reduced to 27 hinge coordinates',
                       'mesh/texture paths resolved, source inertias and balanceinertia retained',
                       'runtime disables contact and uses sampled external PD + Euler'],
              warning='Stand identification variant, NOT original floating/contact locomotion scene.')
    metadata.write_text(json.dumps(info,indent=2)+'\n')
    return output
