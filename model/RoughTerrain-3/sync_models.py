#!/usr/bin/env python3
"""Sync quad URDF, then biped/quad MJCF from the authoritative biped URDF.

Requires numpy. Changes are applied with apply_patch; existing MJCF scenes,
actuator order, sensors and sites are preserved. Joint velocity limits are
stored as custom numeric metadata, NOT enforced by MuJoCo.
"""
import argparse
import copy
import difflib
import math
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT = Path(__file__).resolve().parent
C = np.array([[0., 0., 1.], [0., 1., 0.], [-1., 0., 0.]])


def fmt(v):
    return ' '.join(format(0. if abs(float(x)) < 1e-14 else float(x), '.15g') for x in np.asarray(v).flat)


def vec(s):
    return np.fromstring(s, sep=' ')


def rotation(v):
    r, p, y = v
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([[cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
                     [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr], [-sp, cp*sr, cp*cr]])


def rpy(m):
    p = math.atan2(-m[2, 0], math.hypot(m[0, 0], m[1, 0]))
    if abs(math.cos(p)) > 1e-10:
        return np.array([math.atan2(m[2, 1], m[2, 2]), p, math.atan2(m[1, 0], m[0, 0])])
    return np.array([math.atan2(-m[1, 2], m[1, 1]), p, 0.])


def quat(m):
    r, p, y = rpy(m) / 2
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy, cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy]


def origin(element):
    o = element.find('origin')
    if o is None:
        return np.zeros(3), np.eye(3)
    return vec(o.get('xyz', '0 0 0')), rotation(vec(o.get('rpy', '0 0 0')))


def set_origin(element, p, r):
    o = element.find('origin')
    if o is None:
        o = ET.SubElement(element, 'origin')
    o.attrib.update(xyz=fmt(p), rpy=fmt(rpy(r)))


def inertia(element):
    a = element.find('inertia').attrib
    return np.array([[float(a['ixx']), float(a['ixy']), float(a['ixz'])],
                     [float(a['ixy']), float(a['iyy']), float(a['iyz'])],
                     [float(a['ixz']), float(a['iyz']), float(a['izz'])]])


def set_inertia(element, m):
    for key, i, j in [('ixx',0,0), ('iyy',1,1), ('izz',2,2), ('ixy',0,1), ('ixz',0,2), ('iyz',1,2)]:
        element.find('inertia').set(key, fmt([m[i, j]]))


def load(path, comments=False):
    return ET.parse(path, parser=ET.XMLParser(target=ET.TreeBuilder(insert_comments=comments))).getroot()


def apply(path, root):
    ET.indent(root, space='  ')
    result = '<?xml version="1.0" encoding="utf-8"?>\n' + ET.tostring(root, encoding='unicode') + '\n'
    original = path.read_text()
    if result == original:
        print(path.name, 'unchanged')
        return
    hunks = list(difflib.unified_diff([x+'\n' for x in original.splitlines()], [x+'\n' for x in result.splitlines()]))[2:]
    hunks = ['@@\n' if line.startswith('@@ ') else line for line in hunks]
    patch = '*** Begin Patch\n*** Update File: ' + str(path) + '\n' + ''.join(hunks) + '*** End Patch\n'
    subprocess.run(['apply_patch'], input=patch, text=True, check=True)


def quadrupedal(source):
    out = copy.deepcopy(source)
    out.insert(0, ET.Comment(' Generated from bipedal URDF: C=Ry(+90deg), p_new=C*p_old; joint R_new=C*R_old*C.T; geometry R_new=C*R_old; I_new=C*I_old*C.T. '))
    for link in out.findall('link'):
        for tag in ('visual', 'collision'):
            for g in link.findall(tag):
                p, r = origin(g)
                set_origin(g, C @ p, C @ r)
        item = link.find('inertial')
        if item is not None:
            p, r = origin(item)
            set_inertia(item, C @ r @ inertia(item) @ r.T @ C.T)
            set_origin(item, C @ p, np.eye(3))
    for j in out.findall('joint'):
        p, r = origin(j)
        set_origin(j, C @ p, C @ r @ C.T)
        a = j.find('axis')
        if a is not None:
            a.set('xyz', fmt(C @ vec(a.get('xyz'))))
    return out


def validate_frames(source, target):
    for l, q in zip(source.findall('link'), target.findall('link')):
        assert l.get('name') == q.get('name')
        for tag in ('visual', 'collision'):
            assert len(l.findall(tag)) == len(q.findall(tag))
            for a, b in zip(l.findall(tag), q.findall(tag)):
                p, r = origin(a); pq, rq = origin(b)
                np.testing.assert_allclose(pq, C @ p, atol=1e-12)
                np.testing.assert_allclose(rq, C @ r, atol=1e-12)
                assert [(x.tag,x.attrib) for x in a.find('geometry')] == [(x.tag,x.attrib) for x in b.find('geometry')]
        a, b = l.find('inertial'), q.find('inertial')
        assert (a is None) == (b is None)
        if a is not None:
            p, r = origin(a); pq, rq = origin(b)
            np.testing.assert_allclose(pq, C @ p, atol=1e-12)
            np.testing.assert_allclose(rq @ inertia(b) @ rq.T, C @ r @ inertia(a) @ r.T @ C.T, atol=1e-12)
            assert a.find('mass').attrib == b.find('mass').attrib
    for a, b in zip(source.findall('joint'), target.findall('joint')):
        assert a.attrib == b.attrib
        p, r = origin(a); pq, rq = origin(b)
        np.testing.assert_allclose(pq, C @ p, atol=1e-12)
        np.testing.assert_allclose(rq, C @ r @ C.T, atol=1e-12)
        aa, ab = a.find('axis'), b.find('axis')
        assert (aa is None) == (ab is None)
        if aa is not None:
            np.testing.assert_allclose(vec(ab.get('xyz')), C @ vec(aa.get('xyz')), atol=1e-12)
        for tag in ('limit', 'parent', 'child', 'mimic'):
            aa, bb = a.find(tag), b.find(tag)
            assert (aa.attrib if aa is not None else None) == (bb.attrib if bb is not None else None)
    print('URDF frame, geometry, inertia, limits and mimic validation passed')


def mjcf(source, path):
    scene = load(path, comments=True)
    asset = scene.find('asset')
    world = scene.find('worldbody')
    old_base = world.find("body[@name='base_link']")
    old_bodies = {b.get('name'): b for b in old_base.iter('body')}
    meshes = {m.get('name'): m for m in asset.findall('mesh')}
    joints = {j.get('name'): j for j in source.findall('joint')}
    links = {l.get('name'): l for l in source.findall('link')}
    by_parent = {}
    for j in joints.values():
        by_parent.setdefault(j.find('parent').get('link'), []).append(j)

    def build(name, incoming=None):
        old = old_bodies.get(name)
        attrs = dict(old.attrib) if old is not None else {'name': name}
        if incoming is not None:
            p, r = origin(incoming)
            attrs.update(pos=fmt(p), quat=fmt(quat(r)))
        b = ET.Element('body', attrs)
        if incoming is None:
            for free in old.findall('freejoint'):
                b.append(copy.deepcopy(free))
        elif incoming.get('type') != 'fixed':
            typ = incoming.get('type')
            a = {'name': incoming.get('name'), 'pos': '0 0 0', 'axis': incoming.find('axis').get('xyz'),
                 'type': 'slide' if typ == 'prismatic' else 'hinge', 'limited': 'false' if typ == 'continuous' else 'true'}
            lim = incoming.find('limit')
            if typ != 'continuous':
                a['range'] = lim.get('lower') + ' ' + lim.get('upper')
            ET.SubElement(b, 'joint', a)
        l = links[name]
        it = l.find('inertial')
        # Pure reference-frame links intentionally have no mass/inertia.
        if it is not None:
            p, r = origin(it)
            m = r @ inertia(it) @ r.T
            ET.SubElement(b, 'inertial', pos=fmt(p), mass=it.find('mass').get('value'),
                          fullinertia=fmt([m[0,0],m[1,1],m[2,2],m[0,1],m[0,2],m[1,2]]))
        if old is not None:
            for site in old.findall('site'):
                b.append(copy.deepcopy(site))
        for tag, cls in [('visual', 'visualgeom'), ('collision', 'contactgeom')]:
            for index, item in enumerate(l.findall(tag)):
                shape = list(item.find('geometry'))[0]
                p, r = origin(item)
                a = {'name': name+'_'+tag+('' if index == 0 else '_'+str(index+1)),
                     'type': shape.tag, 'pos': fmt(p), 'quat': fmt(quat(r)), 'class': cls}
                if shape.tag == 'mesh':
                    filename = Path(shape.get('filename')).name
                    if filename == 'base_link.STL':
                        filename = 'base_link.obj'
                    meshname = name + '_' + tag + '_mesh'
                    if meshname not in meshes:
                        meshes[meshname] = ET.SubElement(asset, 'mesh', name=meshname)
                    meshes[meshname].set('file', 'meshes/'+filename)
                    if shape.get('scale') is not None:
                        meshes[meshname].set('scale', shape.get('scale'))
                    a['mesh'] = meshname
                elif shape.tag == 'box':
                    a['size'] = fmt(vec(shape.get('size')) / 2)
                elif shape.tag == 'cylinder':
                    a['size'] = fmt([float(shape.get('radius')), float(shape.get('length'))/2])
                elif shape.tag == 'sphere':
                    a['size'] = shape.get('radius')
                else:
                    raise ValueError('Unsupported geometry '+shape.tag)
                color = item.find('material/color')
                if tag == 'visual' and color is not None:
                    a['rgba'] = color.get('rgba')
                elif tag == 'collision':
                    a['rgba'] = '0.35 0.45 0.55 0.12'
                ET.SubElement(b, 'geom', a)
        for j in by_parent.get(name, []):
            b.append(build(j.find('child').get('link'), j))
        return b

    new_base = build('base_link')
    at = list(world).index(old_base)
    world.remove(old_base); world.insert(at, new_base)
    used = {g.get('mesh') for g in scene.iter('geom') if g.get('mesh')}
    for mesh in list(asset.findall('mesh')):
        if mesh.get('name') not in used:
            asset.remove(mesh)
    # Remove obsolete comments about the former primitive base workaround.
    for node in list(asset):
        if node.tag is ET.Comment and ('Base STL' in node.text or '944,539' in node.text or '<mesh' in node.text):
            asset.remove(node)
    custom = scene.find('custom')
    if custom is None:
        custom = ET.SubElement(scene, 'custom')
        custom.append(ET.Comment(' URDF velocity limits: rad/s for hinge, m/s for slide; metadata only, controller enforcement required. '))
    for j in joints.values():
        limit = j.find('limit')
        if limit is None:
            continue
        key = 'urdf_velocity_limit_' + j.get('name')
        item = custom.find("numeric[@name='"+key+"']")
        if item is None:
            item = ET.SubElement(custom, 'numeric', name=key)
        item.set('data', limit.get('velocity'))
    for motor in scene.find('actuator'):
        if motor.tag != 'motor':
            continue
        effort = float(joints[motor.get('joint')].find('limit').get('effort'))
        motor.attrib.update(gear='1', ctrllimited='true', ctrlrange=fmt([-effort,effort]),
                            forcelimited='true', forcerange=fmt([-effort,effort]))
    eq = scene.find('equality')
    for j in joints.values():
        mimic = j.find('mimic')
        if mimic is not None:
            item = eq.find("joint[@joint1='"+j.get('name')+"']")
            if item is None:
                item = ET.SubElement(eq, 'joint', name=j.get('name')+'_mimic')
            item.attrib.update(joint1=j.get('name'), joint2=mimic.get('joint'),
                               polycoef=mimic.get('offset','0')+' '+mimic.get('multiplier','1')+' 0 0 0')
    keyframes = scene.find('keyframe')
    if keyframes is None:
        keyframes = ET.SubElement(scene, 'keyframe')
    nominal = keyframes.find("key[@name='nominal']")
    if nominal is None:
        positions = list(vec(new_base.get('pos', '0 0 0'))) + list(vec(new_base.get('quat', '1 0 0 0')))
        # MJCF qpos follows body-tree traversal, not the flat URDF joint order.
        for joint in new_base.iter('joint'):
            value = 0.0
            if joint.get('limited') == 'true':
                lo, hi = vec(joint.get('range'))
                value = min(max(value, lo), hi)
            positions.append(value)
        ET.SubElement(keyframes, 'key', name='nominal', qpos=fmt(positions))
    return scene


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['urdf','bipedal','quadrupedal'])
    args = parser.parse_args()
    source = load(ROOT/'urdf/RoughTerrain-3_bipedal.urdf')
    if args.stage == 'urdf':
        quad = quadrupedal(source)
        validate_frames(source, quad)
        apply(ROOT/'urdf/RoughTerrain-3_quadrupedal.urdf', quad)
    else:
        if args.stage == 'quadrupedal':
            source = load(ROOT/'urdf/RoughTerrain-3_quadrupedal.urdf')
        path = ROOT/('RoughTerrain-3_'+args.stage+'.xml')
        apply(path, mjcf(source, path))


if __name__ == '__main__':
    main()
