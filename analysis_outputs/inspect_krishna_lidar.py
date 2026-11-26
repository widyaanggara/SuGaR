import json
import struct
import zlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree

source = Path(r'C:\Users\anggara\Downloads\dewa_krishna.ply')
out = Path(__file__).parent / 'krishna_lidar'
out.mkdir(exist_ok=True)
types = {'double': '<f8', 'float': '<f4', 'ushort': '<u2', 'uchar': 'u1', 'char': 'i1', 'int': '<i4'}
with source.open('rb') as f:
    header, properties = [], []
    while True:
        line = f.readline().decode('ascii').strip()
        header.append(line)
        if line.startswith('element vertex '):
            count = int(line.split()[-1])
        if line.startswith('property '):
            _, kind, name = line.split()
            properties.append((name, types[kind]))
        if line == 'end_header':
            break
    data = np.fromfile(f, dtype=np.dtype(properties), count=count)
xyz = np.column_stack([data[n] for n in ('x', 'y', 'z')])
valid = np.isfinite(xyz).all(axis=1)
p = np.unique(xyz[valid], axis=0)
dist, _ = cKDTree(p).query(p, k=17, workers=-1)
nn = dist[:, 1]
local = dist[:, 1:].mean(axis=1)
stats = {
    'source': str(source), 'unit': 'unknown; values below are file-coordinate units',
    'vertices': len(xyz), 'nonfinite': int((~valid).sum()),
    'duplicate_xyz': int(valid.sum()-len(p)), 'header': header,
    'minimum_xyz': p.min(axis=0).tolist(), 'maximum_xyz': p.max(axis=0).tolist(),
    'extent_xyz': np.ptp(p, axis=0).tolist(),
    'nearest_neighbor_percentiles': dict(zip(['p5','p25','p50','p75','p95','p99','max'], np.percentile(nn, [5,25,50,75,95,99,100]).tolist())),
    'mean_16_neighbor_distance_percentiles': np.percentile(local, [50,95,99,100]).tolist(),
    'statistical_outlier_candidates_mean16_plus3std': int((local > local.mean()+3*local.std()).sum()),
    'rgb_range': {n: [int(data[n].min()),int(data[n].max())] for n in ('red','green','blue')},
}
(out/'statistics.json').write_text(json.dumps(stats, indent=2), encoding='utf-8')
# Orthographic density projections, left to right: XY, XZ, YZ.
# Log count shading makes sparse boundary points visible without inferring a surface.
canvas = np.full((850, 2100, 3), 248, dtype=np.uint8)
for panel, (a,b) in enumerate([(0,1),(0,2),(1,2)]):
    lo = xyz[valid][:,[a,b]].min(axis=0)
    span = np.ptp(xyz[valid][:,[a,b]],axis=0)
    scale = min(640/span[0],790/span[1])
    uv = ((xyz[valid][:,[a,b]]-lo)*scale).astype(int)
    offset = ((np.array([700,850])-span*scale)/2).astype(int)
    xx = panel*700+offset[0]+uv[:,0]
    yy = 849-offset[1]-uv[:,1]
    counts = np.zeros((850,2100),dtype=np.int32)
    np.add.at(counts,(yy,xx),1)
    mask = counts>0
    shade = np.minimum(np.log1p(counts[mask])/np.log(20),1)
    canvas[mask] = np.column_stack([30+70*(1-shade),65+70*(1-shade),90+70*(1-shade)]).astype(np.uint8)
def chunk(kind, payload):
    return struct.pack('>I',len(payload))+kind+payload+struct.pack('>I',zlib.crc32(kind+payload)&0xffffffff)
raw = b''.join(b'\0'+row.tobytes() for row in canvas)
(out/'projections_xy_xz_yz.png').write_bytes(b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',2100,850,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b''))
print(json.dumps({k:v for k,v in stats.items() if k!='header'}, indent=2))
