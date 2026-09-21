// 修改 for 循环内部的逻辑
for (faceIdx, indices) in faceIndices.enumerated() {
    // 🔥 决定当前面的纹理 ID
    let texID: UInt32 = faceIdx < 3 ? 0 : 1
    
    let v0 = rotate(vertices3D[indices[0]])
    let v1 = rotate(vertices3D[indices[1]])
    let v2 = rotate(vertices3D[indices[2]])
    let v3 = rotate(vertices3D[indices[3]])
    
    let n = rotate(faceNormals[faceIdx])
    
    let (p0, z0) = project(v0)
    let (p1, z1) = project(v1)
    let (p2, z2) = project(v2)
    let (p3, z3) = project(v3)
    
    let color = colors[faceIdx]
    let uvs = faceUVs[faceIdx]
    
    gal.drawTriangle(
        p0: p0, p1: p1, p2: p2,
        color: color,
        z0: z0, z1: z1, z2: z2,
        textureID: texID, // 🔥 传入 textureID
        uv0: uvs[0], uv1: uvs[1], uv2: uvs[2],
        n0: n, n1: n, n2: n
    )
    
    gal.drawTriangle(
        p0: p0, p1: p2, p2: p3,
        color: color,
        z0: z0, z1: z2, z2: z3,
        textureID: texID, // 🔥 传入 textureID
        uv0: uvs[0], uv1: uvs[2], uv2: uvs[3],
        n0: n, n1: n, n2: n
    )
}