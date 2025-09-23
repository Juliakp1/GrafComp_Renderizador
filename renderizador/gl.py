#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# pylint: disable=invalid-name

"""
Biblioteca Gráfica / Graphics Library.

Desenvolvido por: <SEU NOME AQUI>
Disciplina: Computação Gráfica
Data: <DATA DE INÍCIO DA IMPLEMENTAÇÃO>
"""

import time         # Para operações com tempo
import gpu          # Simula os recursos de uma GPU
import math         # Funções matemáticas
import numpy as np  # Biblioteca do Numpy

class GL:
    """Classe que representa a biblioteca gráfica (Graphics Library)."""

    width = 800   # largura da tela
    height = 600  # altura da tela
    near = 0.01   # plano de corte próximo
    far = 1000    # plano de corte distante
    top = 0.004142      # coordenada top do plano de corte
    bottom = 0   
    right = 0.005522    # coordenada right do plano de corte
    left = 0
    fovDiag = 60 
    fovy = 60   # campo de visão vertical

    VIEW = []
    STACK = [np.eye(4)]
    ZBUFFER = np.full((height, width), float('inf'))
    LIGHTS = []

    @staticmethod
    def setup(width, height, near=0.01, far=1000):
        """Definr parametros para câmera de razão de aspecto, plano próximo e distante."""
        GL.width = width
        GL.height = height
        GL.near = near
        GL.far = far

        print("width = {0}, height = {1}, near = {2}, far = {3}".format(width, height, near, far))

    # --------------------------------------------------------------- #

    @staticmethod
    def quaternionToRotationMatrix(q):
        """Converts a quaternion into a 3x3 rotation matrix."""
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm
        else:
            q = np.array([0, 0, 0, 1])

        x, y, z, w = q
        
        return np.array([
            [1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y, 0],
            [2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x, 0],
            [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2, 0],
            [0, 0, 0, 1]
        ])
    
    @staticmethod
    def translationMatrix(translation):
        T = np.eye(4)
        T[0, 3] = translation[0]
        T[1, 3] = translation[1]
        T[2, 3] = translation[2]
        return T

    @staticmethod
    def perspectiveTransformMatrix(near, far, right, top):
        return np.array([
            [near/right, 0, 0, 0],
            [0, near/top, 0, 0],
            [0, 0, -(far + near)/(far - near), -(2 * far * near)/(far - near)],
            [0, 0, -1, 0]
        ])
    
    @staticmethod
    def viewportTransformMatrix(width, height):
        return np.array([
            [width / 2, 0, 0, width / 2],
            [0, -(height / 2), 0, height / 2],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # --------------------------------------------------------------- #

    @staticmethod
    def polypoint2D(point, colors):
        """Função usada para renderizar Polypoint2D."""
        color = [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]
        for i in range(0, len(point), 2):
            x = int(point[i])
            y = int(point[i + 1])
            gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    # --------------------------------------------------------------- #

    @staticmethod
    def polyline2D(lineSegments, colors):
        """Função usada para renderizar Polyline2D."""

        color = [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]
        for i in range(0, len(lineSegments)-2, 2):

            x1, y1 = lineSegments[i:i+2]
            x2, y2 = lineSegments[i+2:i+4]

            dx = x2 - x1
            dy = y2 - y1
            steps = max(abs(dx), abs(dy))

            x_increment = dx / steps
            y_increment = dy / steps

            x = x1
            y = y1
            for j in range(int(steps) + 1):
                if x >= 0 and x < GL.width and y >= 0 and y < GL.height:
                    gpu.GPU.draw_pixel([int(x), int(y)], gpu.GPU.RGB8, color)
                x += x_increment
                y += y_increment

    # --------------------------------------------------------------- #

    @staticmethod
    def circle2D(radius, colors):
        """Função usada para renderizar Circle2D."""

        color = [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]
        for angle in np.arange(0, 2 * math.pi, 0.01):
            x = int(radius * math.cos(angle))
            y = int(radius * math.sin(angle))
            if x >= 0 and x < GL.width and y >= 0 and y < GL.height:
                gpu.GPU.draw_pixel([x, y], gpu.GPU.RGB8, color)

    # --------------------------------------------------------------- #

    @staticmethod
    def triangleSet2D(vertices, colors, z_values=[1,1,1], transparency=1, texture=None, texture_coords=None, normal=None):
        """Função usada para renderizar TriangleSet2D."""

        def test_point(x, y, x1, x2, y1, y2):
            t1 = x - x1
            t2 = y - y1
            t3 = x2 - x1
            t4 = y2 - y1
            final = t1 * t4 - t2 * t3
            return 0 <= final
        
        grid_size = 2 # 4 samples per pixel
        num_samples = grid_size * grid_size

        for i in range(0, len(vertices), 6):
            x1, y1 = vertices[i], vertices[i + 1]
            x2, y2 = vertices[i + 2], vertices[i + 3]
            x3, y3 = vertices[i + 4], vertices[i + 5]

            inv_z1 = 1.0 / z_values[0]
            inv_z2 = 1.0 / z_values[1]
            inv_z3 = 1.0 / z_values[2]

            # bounding box 
            min_x = int(min(x1, x2, x3))
            max_x = int(max(x1, x2, x3))
            min_y = int(min(y1, y2, y3))
            max_y = int(max(y1, y2, y3))

            for y_pixel in range(max(0, min_y), min(GL.height, max_y + 1)):
                for x_pixel in range(max(0, min_x), min(GL.width, max_x + 1)):
                    
                    current_depth = GL.ZBUFFER[y_pixel][x_pixel]
                    
                    for sy in range(grid_size):
                        for sx in range(grid_size):
                            
                            # sx, sy = sub-sample coordinate
                            sub_x = x_pixel + (sx + 0.5) / grid_size
                            sub_y = y_pixel + (sy + 0.5) / grid_size
                            
                            if (
                                test_point(sub_x, sub_y, x1, x2, y1, y2) and
                                test_point(sub_x, sub_y, x2, x3, y2, y3) and
                                test_point(sub_x, sub_y, x3, x1, y3, y1)
                            ):
                                
                                totalArea = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
                                alpha = ((x2 - sub_x) * (y3 - sub_y) - (x3 - sub_x) * (y2 - sub_y)) / totalArea
                                beta = ((sub_x - x1) * (y3 - y1) - (x3 - x1) * (sub_y - y1)) / totalArea
                                gamma = 1.0 - alpha - beta
                                
                                inv_z_sub = alpha * inv_z1 + beta * inv_z2 + gamma * inv_z3
                                z_sub = 1.0 / inv_z_sub

                                if z_sub < current_depth:
                                    
                                    # Texture mapping
                                    if texture is not None and texture_coords is not None:
                                        u1, v1 = texture_coords[i], texture_coords[i + 1]
                                        u2, v2 = texture_coords[i + 2], texture_coords[i + 3]
                                        u3, v3 = texture_coords[i + 4], texture_coords[i + 5]

                                        u_sub = (alpha * u1 * inv_z1 + beta * u2 * inv_z2 + gamma * u3 * inv_z3) * z_sub
                                        v_sub = (alpha * v1 * inv_z1 + beta * v2 * inv_z2 + gamma * v3 * inv_z3) * z_sub

                                        tex_x = int(u_sub * (len(texture[0]) - 1))
                                        tex_y = int((1 - v_sub) * (len(texture) - 1)) # Invert v coordinate

                                        tex_x = max(0, min(len(texture[0]) - 1, tex_x))
                                        tex_y = max(0, min(len(texture) - 1, tex_y))

                                        final_color = texture[tex_x][tex_y]
                                        final_color = final_color[:3]  # Ignore alpha if present

                                    # -------------------------------- #

                                    # Color per vertex
                                    elif len(colors) > 8:
                                        z1, r1, g1, b1 = z_values[0], colors[0], colors[1], colors[2]
                                        z2, r2, g2, b2 = z_values[1], colors[3], colors[4], colors[5]
                                        z3, r3, g3, b3 = z_values[2], colors[6], colors[7], colors[8]

                                        r_over_z = alpha * (r1 / z1) + beta * (r2 / z2) + gamma * (r3 / z3)
                                        g_over_z = alpha * (g1 / z1) + beta * (g2 / z2) + gamma * (g3 / z3)
                                        b_over_z = alpha * (b1 / z1) + beta * (b2 / z2) + gamma * (b3 / z3)

                                        final_r = r_over_z * z_sub
                                        final_g = g_over_z * z_sub
                                        final_b = b_over_z * z_sub

                                        new_color = [int(final_r), int(final_g), int(final_b)]

                                        if transparency < 1:
                                            existing_color = gpu.GPU.read_pixel([x_pixel, y_pixel], gpu.GPU.RGB8)
                                            final_color = [
                                                int(existing_color[0] * (1 - transparency) + new_color[0] * transparency),
                                                int(existing_color[1] * (1 - transparency) + new_color[1] * transparency),
                                                int(existing_color[2] * (1 - transparency) + new_color[2] * transparency)
                                            ]
                                        else:
                                            final_color = new_color

                                    # -------------------------------- #

                                    # Single color
                                    else:
                                        if "emissiveColor" in colors: # corrects ones that werent unpacked
                                            new_color = [
                                                colors["emissiveColor"][0] * 255,
                                                colors["emissiveColor"][1] * 255,
                                                colors["emissiveColor"][2] * 255]
                                        else:
                                            new_color = [colors[0], colors[1], colors[2]]
                                            
                                        if transparency < 1:
                                            existing_color = gpu.GPU.read_pixel([x_pixel, y_pixel], gpu.GPU.RGB8)
                                            final_color = [
                                                int(existing_color[0] * (1 - transparency) + new_color[0] * transparency),
                                                int(existing_color[1] * (1 - transparency) + new_color[1] * transparency),
                                                int(existing_color[2] * (1 - transparency) + new_color[2] * transparency)
                                            ]
                                        else:
                                            final_color = new_color

                                    # -------------------------------- #

                                    # Lighting
                                    if normal is not None and GL.LIGHTS:
                                        # TODO
                                        pass
                                    
                                    # -------------------------------- #

                                    gpu.GPU.draw_pixel([x_pixel, y_pixel], gpu.GPU.RGB8, final_color)
                                    GL.ZBUFFER[y_pixel][x_pixel] = z_sub

    # --------------------------------------------------------------- #

    @staticmethod
    def triangleSet(point, colors, transparency=1, texture=None, texture_coords=None):
        """Função usada para renderizar TriangleSet."""

        perspMatrix = GL.perspectiveTransformMatrix(GL.near, GL.far, GL.right, GL.top)
        viewportMatrix = GL.viewportTransformMatrix(GL.width, GL.height)

        if "transparency" in colors and colors["transparency"] > 0:
            transparency = colors["transparency"]
        
        for i in range(0, len(point), 9):
            x1, y1, z1 = point[i:i+3]
            x2, y2, z2 = point[i+3:i+6]
            x3, y3, z3 = point[i+6:i+9]

            # Calculates the normal vector of the triangle
            v1 = np.array([x2 - x1, y2 - y1, z2 - z1])
            v2 = np.array([x3 - x1, y3 - y1, z3 - z1])
            normal = np.cross(v1, v2)

            # Homogenous coordinates
            p1 = np.array([x1, y1, z1, 1])
            p2 = np.array([x2, y2, z2, 1])
            p3 = np.array([x3, y3, z3, 1])

            # Apply the Model, View, and Projection matrices to each vertex.
            proj_p1 = perspMatrix @ GL.VIEW @ GL.STACK[-1] @ p1
            proj_p2 = perspMatrix @ GL.VIEW @ GL.STACK[-1] @ p2
            proj_p3 = perspMatrix @ GL.VIEW @ GL.STACK[-1] @ p3

            originalZ = [proj_p1[2], proj_p2[2], proj_p3[2]] # Store original z values for z buffer

            # Divide by w (Perspective Divide)
            p1_clip = proj_p1 / (proj_p1[3] if proj_p1[3] != 0 else 0.1)
            p2_clip = proj_p2 / (proj_p2[3] if proj_p2[3] != 0 else 0.1)
            p3_clip = proj_p3 / (proj_p3[3] if proj_p3[3] != 0 else 0.1)

            # Apply the viewport transformation
            p1_viewport = viewportMatrix @ p1_clip
            p2_viewport = viewportMatrix @ p2_clip
            p3_viewport = viewportMatrix @ p3_clip

            # Final 2D coordinates
            projVertices = np.array([
                p1_viewport[0], p1_viewport[1],
                p2_viewport[0], p2_viewport[1],
                p3_viewport[0], p3_viewport[1]
            ])

            GL.triangleSet2D(projVertices, 
                             colors, 
                             z_values=originalZ, 
                             transparency=transparency, 
                             texture=texture, 
                             texture_coords=texture_coords,
                             normal=normal)

    # --------------------------------------------------------------- #

    @staticmethod
    def viewpoint(position, orientation, fieldOfView):
        """Função usada para renderizar (na verdade coletar os dados) de Viewpoint."""
        print("Viewpoint : ", end='')
        print("position = {0} ".format(position), end='')
        print("orientation = {0} ".format(orientation), end='')
        print("fieldOfView = {0} ".format(fieldOfView))

        aspect = GL.width / GL.height
        GL.fovDiag = fieldOfView

        GL.top = GL.near * math.tan(fieldOfView / 2)
        GL.bottom = -GL.top
        GL.right = GL.top * aspect
        GL.left = -GL.right

        print("Updated fovy = {0}, top = {1}, bottom = {2}, right = {3}, left = {4}".format(GL.fovy, GL.top, GL.bottom, GL.right, GL.left))

        eye = np.array(position)
        at = eye + np.array([0, 0, -1])
        up = np.array([0, 1, 0])

        w = (at - eye) / np.linalg.norm(at - eye)
        u = np.cross(w, up) / np.linalg.norm(np.cross(w, up))
        v = np.cross(u, w) / np.linalg.norm(np.cross(u, w))

        GL.VIEW = np.array([
            [u[0], u[1], u[2], -np.dot(u, eye)],
            [v[0], v[1], v[2], -np.dot(v, eye)],
            [-w[0], -w[1], -w[2], np.dot(w, eye)],
            [0, 0, 0, 1]
        ])

    # --------------------------------------------------------------- #
    
    @staticmethod
    def transform_in(translation, scale, rotation):
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""

        def scaleMatrix(scale):
            S = np.eye(4)
            S[0, 0] = scale[0]
            S[1, 1] = scale[1]
            S[2, 2] = scale[2]
            return S

        lastMatrix = GL.STACK[-1]

        x, y, z, angle = rotation
        rotationQuaternion = np.array([x * math.sin(angle / 2), y * math.sin(angle / 2), z * math.sin(angle / 2), math.cos(angle / 2)])
        
        allTransforms = GL.translationMatrix(translation) @ GL.quaternionToRotationMatrix(rotationQuaternion) @ scaleMatrix(scale)
    
        GL.STACK.append(lastMatrix @ allTransforms)

    # --------------------------------------------------------------- #

    @staticmethod
    def transform_out():
        """Função usada para renderizar (na verdade coletar os dados) de Transform."""
        GL.STACK.pop()

    # --------------------------------------------------------------- #

    @staticmethod
    def triangleStripSet(point, stripCount, colors):
        """Função usada para renderizar TriangleStripSet."""

        print("\nStrip Length : {0} - Color: {1}".format(len(point), [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]))

        pointIndex = 0
        for strip_len in stripCount:
            swapDirection = False
            
            for i in range(strip_len - 2):
                
                x1, y1, z1 = point[pointIndex + i*3 : pointIndex + i*3 + 3]
                x2, y2, z2 = point[pointIndex + i*3 + 3 : pointIndex + i*3 + 6]
                x3, y3, z3 = point[pointIndex + i*3 + 6 : pointIndex + i*3 + 9]
                
                if swapDirection:
                    GL.triangleSet([x2, y2, z2, x1, y1, z1, x3, y3, z3], colors)
                else:
                    GL.triangleSet([x1, y1, z1, x2, y2, z2, x3, y3, z3], colors)
                
                swapDirection = not swapDirection
                print(i, end=' ', flush=True)
            
            pointIndex += strip_len * 3

    # --------------------------------------------------------------- #

    @staticmethod
    def indexedTriangleStripSet(point, index, colors):
        """Função usada para renderizar IndexedTriangleStripSet."""

        print("\nIndexed Length : {0} - Color: {1}".format(len(index), [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]))

        swapDirection = False
        p = 0
        while p < len(index) - 2:
            
            if index[p] == -1 or index[p+1] == -1 or index[p+2] == -1:
                p += 1
                swapDirection = False
                continue

            i = index[p]
            j = index[p+1]
            k = index[p+2]

            x1, y1, z1 = point[i*3:i*3+3]
            x2, y2, z2 = point[j*3:j*3+3]
            x3, y3, z3 = point[k*3:k*3+3]

            if swapDirection:
                GL.triangleSet([x2, y2, z2, x1, y1, z1, x3, y3, z3], colors)
            else:
                GL.triangleSet([x1, y1, z1, x2, y2, z2, x3, y3, z3], colors)

            swapDirection = not swapDirection
            p += 1
            print(p, end=' ', flush=True)

    # --------------------------------------------------------------- #

    @staticmethod
    def indexedFaceSet(coord, coordIndex, colorPerVertex, color, colorIndex,
                       texCoord, texCoordIndex, colors, current_texture):
        """Função usada para renderizar IndexedFaceSet."""

        print("\nFaces Length : {0}".format(len(coordIndex)), flush=True)

        transparency = 1.0
        if "transparency" in colors and colors["transparency"] > 0:
            transparency = colors["transparency"]

        textureImage = None
        if current_texture:
            textureImage = gpu.GPU.load_texture(current_texture[0])

        # Splits the lists
        list_of_fans = []
        list_of_colors = []
        current_fan_indices = []
        current_colors = []
        list_of_texture_coords = []
        current_texture_coords = []

        for i in range(len(coordIndex)):
            if coordIndex[i] == -1:
                list_of_fans.append(current_fan_indices)
                list_of_colors.append(current_colors)
                list_of_texture_coords.append(current_texture_coords)
                current_fan_indices = []
                current_colors = []
                current_texture_coords = []
            else:
                current_fan_indices.append(coordIndex[i])
                if colorPerVertex and color:
                    current_colors.append(colorIndex[i])
                if textureImage is not None and texCoord and texCoordIndex:
                    current_texture_coords.append(texCoordIndex[i])
                continue

        # -------------------------------- #

        for i in range(len(list_of_fans)):
            current_fan_indices = list_of_fans[i]
            current_colors = list_of_colors[i]
            current_texture_coords = list_of_texture_coords[i]
            anchor_idx = current_fan_indices[0]
            swapDirection = False

            for j in range(1, len(current_fan_indices) - 1):

                # Triangle vertices
                p1_idx = anchor_idx
                p2_idx = current_fan_indices[j]
                p3_idx = current_fan_indices[j + 1]
                    
                x1, y1, z1 = coord[p1_idx * 3 : p1_idx * 3 + 3]
                x2, y2, z2 = coord[p2_idx * 3 : p2_idx * 3 + 3]
                x3, y3, z3 = coord[p3_idx * 3 : p3_idx * 3 + 3]

                # -------------------------------- #

                triangle_coords = []
                texture_coords = None
                final_color = []
                
                # Triangle with texture
                if textureImage is not None:

                    if current_texture_coords != []:
                        t1_idx = current_texture_coords[0]
                        t2_idx = current_texture_coords[j]
                        t3_idx = current_texture_coords[j + 1]

                        u1, v1 = texCoord[t1_idx * 2 : t1_idx * 2 + 2]
                        u2, v2 = texCoord[t2_idx * 2 : t2_idx * 2 + 2]
                        u3, v3 = texCoord[t3_idx * 2 : t3_idx * 2 + 2]
                    else:
                        u1, v1, u2, v2, u3, v3 = texCoord # somehow an actual thing that happens

                    if not swapDirection:
                        triangle_coords = [x1, y1, z1, x2, y2, z2, x3, y3, z3]
                        texture_coords = [u1, v1, u2, v2, u3, v3]
                    else:
                        triangle_coords = [x3, y3, z3, x2, y2, z2, x1, y1, z1]
                        texture_coords = [u3, v3, u2, v2, u1, v1]

                    final_color = [0, 0, 0]
                    
                # -------------------------------- #

                # Triangle multi colors
                elif colorPerVertex and color: 
                    c1_idx = current_colors[0]
                    c2_idx = current_colors[j]
                    c3_idx = current_colors[j + 1]

                    color1 = color[c1_idx * 3 : c1_idx * 3 + 3]
                    color2 = color[c2_idx * 3 : c2_idx * 3 + 3]
                    color3 = color[c3_idx * 3 : c3_idx * 3 + 3]

                    if not swapDirection:
                        triangle_coords = [x1, y1, z1, x2, y2, z2, x3, y3, z3]
                        final_color = [
                            color1[0]*255, color1[1]*255, color1[2]*255,
                            color2[0]*255, color2[1]*255, color2[2]*255,
                            color3[0]*255, color3[1]*255, color3[2]*255
                        ]
                    else:
                        triangle_coords = [x3, y3, z3, x2, y2, z2, x1, y1, z1]
                        final_color = [
                            color3[0]*255, color3[1]*255, color3[2]*255,
                            color2[0]*255, color2[1]*255, color2[2]*255,
                            color1[0]*255, color1[1]*255, color1[2]*255
                        ]

                # -------------------------------- #

                else:
                    triangle_coords = [x1, y1, z1, x2, y2, z2, x3, y3, z3]
                    final_color = [colors["emissiveColor"][0]*255, colors["emissiveColor"][1]*255, colors["emissiveColor"][2]*255]

                GL.triangleSet(triangle_coords, final_color, transparency=transparency, texture=textureImage, texture_coords=texture_coords)
                swapDirection = not swapDirection
                print(j, end=' ', flush=True)

    # --------------------------------------------------------------- #

    @staticmethod
    def box(size, colors):
        """Função usada para renderizar Boxes."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Box
        # A função box é usada para desenhar paralelepípedos na cena. O Box é centrada no
        # (0, 0, 0) no sistema de coordenadas local e alinhado com os eixos de coordenadas
        # locais. O argumento size especifica as extensões da caixa ao longo dos eixos X, Y
        # e Z, respectivamente, e cada valor do tamanho deve ser maior que zero. Para desenha
        # essa caixa você vai provavelmente querer tesselar ela em triângulos, para isso
        # encontre os vértices e defina os triângulos. Use indexedFaceSet para isso.

        # Define os vértices do box
        sx, sy, sz = size[0] / 2, size[1] / 2, size[2] / 2
        vertices = [
            -sx, -sy, -sz,
            sx, -sy, -sz,
            sx, sy, -sz,
            -sx, sy, -sz,
            -sx, -sy, sz,
            sx, -sy, sz,
            sx, sy, sz,
            -sx, sy, sz
        ]
        indices = [
            0, 1, 2, 3, -1, # back
            4, 5, 6, 7, -1, # front
            0, 4, 7, 3, -1, # left
            1, 5, 6, 2, -1, # right
            3, 2, 6, 7, -1, # top
            0, 1, 5, 4, -1, # bottom
        ]

        GL.indexedFaceSet(vertices, indices, False, None, None, None, None, colors, None)


    # --------------------------------------------------------------- #

    @staticmethod
    def sphere(radius, colors):
        """Função usada para renderizar Esferas."""

        center = [0, 0, 0]
        stacks = 20
        slices = 20
        vertices = []
        indices = []
        for i in range(stacks + 1):
            phi = math.pi * i / stacks
            for j in range(slices + 1):
                theta = 2 * math.pi * j / slices
                x = center[0] + radius * math.sin(phi) * math.cos(theta)
                y = center[1] + radius * math.sin(phi) * math.sin(theta)
                z = center[2] + radius * math.cos(phi)
                vertices.extend([x, y, z])
                
                if i < stacks and j < slices:
                    first = i * (slices + 1) + j
                    second = first + slices + 1
                    indices.extend([first, second, first + 1, second + 1, -1])

        GL.indexedFaceSet(vertices, indices, False, None, None, None, None, colors, None)


    # --------------------------------------------------------------- #

    @staticmethod
    def cone(bottomRadius, height, colors):
        """Função usada para renderizar Cones."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cone
        # A função cone é usada para desenhar cones na cena. O cone é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento bottomRadius especifica o
        # raio da base do cone e o argumento height especifica a altura do cone.
        # O cone é alinhado com o eixo Y local. O cone é fechado por padrão na base.
        # Para desenha esse cone você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cone : bottomRadius = {0}".format(bottomRadius)) # imprime no terminal o raio da base do cone
        print("Cone : height = {0}".format(height)) # imprime no terminal a altura do cone
        print("Cone : colors = {0}".format(colors)) # imprime no terminal as cores

    # --------------------------------------------------------------- #

    @staticmethod
    def cylinder(radius, height, colors):
        """Função usada para renderizar Cilindros."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/geometry3D.html#Cylinder
        # A função cylinder é usada para desenhar cilindros na cena. O cilindro é centrado no
        # (0, 0, 0) no sistema de coordenadas local. O argumento radius especifica o
        # raio da base do cilindro e o argumento height especifica a altura do cilindro.
        # O cilindro é alinhado com o eixo Y local. O cilindro é fechado por padrão em ambas as extremidades.
        # Para desenha esse cilindro você vai precisar tesselar ele em triângulos, para isso
        # encontre os vértices e defina os triângulos.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Cylinder : radius = {0}".format(radius)) # imprime no terminal o raio do cilindro
        print("Cylinder : height = {0}".format(height)) # imprime no terminal a altura do cilindro
        print("Cylinder : colors = {0}".format(colors)) # imprime no terminal as cores

    # --------------------------------------------------------------- #

    @staticmethod
    def navigationInfo(headlight):
        """Características físicas do avatar do visualizador e do modelo de visualização."""

        print("NavigationInfo : headlight = {0}".format(headlight)) 
        if headlight:
            newLight = {
                "type": "DirectionalLight",
                "ambientIntensity": 0.0,
                "color": [1,1,1],
                "intensity": 1,
                "direction": [0,0,-1]
            }
            GL.LIGHTS.append(newLight)   

    # --------------------------------------------------------------- #

    @staticmethod
    def directionalLight(ambientIntensity, color, intensity, direction):
        """Luz direcional ou paralela."""

        print("DirectionalLight : ambientIntensity = {0}".format(ambientIntensity))
        print("DirectionalLight : color = {0}".format(color)) # imprime no terminal
        print("DirectionalLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("DirectionalLight : direction = {0}".format(direction)) # imprime no terminal

        newLight = {
            "type": "DirectionalLight",
            "ambientIntensity": ambientIntensity,
            "color": color,
            "intensity": intensity,
            "direction": direction
        }
        GL.LIGHTS.append(newLight)   

    # --------------------------------------------------------------- #

    @staticmethod
    def pointLight(ambientIntensity, color, intensity, location):
        """Luz pontual."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/lighting.html#PointLight
        # Fonte de luz pontual em um local 3D no sistema de coordenadas local. Uma fonte
        # de luz pontual emite luz igualmente em todas as direções; ou seja, é omnidirecional.
        # Possui os campos básicos ambientIntensity, cor, intensidade. Um nó PointLight ilumina
        # a geometria em um raio de sua localização. O campo do raio deve ser maior ou igual a
        # zero. A iluminação do nó PointLight diminui com a distância especificada.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("PointLight : ambientIntensity = {0}".format(ambientIntensity))
        print("PointLight : color = {0}".format(color)) # imprime no terminal
        print("PointLight : intensity = {0}".format(intensity)) # imprime no terminal
        print("PointLight : location = {0}".format(location)) # imprime no terminal

    # --------------------------------------------------------------- #

    @staticmethod
    def fog(visibilityRange, color):
        """Névoa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/environmentalEffects.html#Fog
        # O nó Fog fornece uma maneira de simular efeitos atmosféricos combinando objetos
        # com a cor especificada pelo campo de cores com base nas distâncias dos
        # vários objetos ao visualizador. A visibilidadeRange especifica a distância no
        # sistema de coordenadas local na qual os objetos são totalmente obscurecidos
        # pela névoa. Os objetos localizados fora de visibilityRange do visualizador são
        # desenhados com uma cor de cor constante. Objetos muito próximos do visualizador
        # são muito pouco misturados com a cor do nevoeiro.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("Fog : color = {0}".format(color)) # imprime no terminal
        print("Fog : visibilityRange = {0}".format(visibilityRange))

    # --------------------------------------------------------------- #

    @staticmethod
    def timeSensor(cycleInterval, loop):
        """Gera eventos conforme o tempo passa."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/time.html#TimeSensor
        # Os nós TimeSensor podem ser usados para muitas finalidades, incluindo:
        # Condução de simulações e animações contínuas; Controlar atividades periódicas;
        # iniciar eventos de ocorrência única, como um despertador;
        # Se, no final de um ciclo, o valor do loop for FALSE, a execução é encerrada.
        # Por outro lado, se o loop for TRUE no final de um ciclo, um nó dependente do
        # tempo continua a execução no próximo ciclo. O ciclo de um nó TimeSensor dura
        # cycleInterval segundos. O valor de cycleInterval deve ser maior que zero.

        # Deve retornar a fração de tempo passada em fraction_changed

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("TimeSensor : cycleInterval = {0}".format(cycleInterval)) # imprime no terminal
        print("TimeSensor : loop = {0}".format(loop))

        # Esse método já está implementado para os alunos como exemplo
        epoch = time.time()  # time in seconds since the epoch as a floating point number.
        fraction_changed = (epoch % cycleInterval) / cycleInterval

        return fraction_changed

    # --------------------------------------------------------------- #

    @staticmethod
    def splinePositionInterpolator(set_fraction, key, keyValue, closed):
        """Interpola não linearmente entre uma lista de vetores 3D."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#SplinePositionInterpolator
        # Interpola não linearmente entre uma lista de vetores 3D. O campo keyValue possui
        # uma lista com os valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantos vetores 3D quanto os
        # quadros-chave no key. O campo closed especifica se o interpolador deve tratar a malha
        # como fechada, com uma transições da última chave para a primeira chave. Se os keyValues
        # na primeira e na última chave não forem idênticos, o campo closed será ignorado.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("SplinePositionInterpolator : set_fraction = {0}".format(set_fraction))
        print("SplinePositionInterpolator : key = {0}".format(key)) # imprime no terminal
        print("SplinePositionInterpolator : keyValue = {0}".format(keyValue))
        print("SplinePositionInterpolator : closed = {0}".format(closed))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0.0, 0.0, 0.0]
        
        return value_changed

    # --------------------------------------------------------------- #

    @staticmethod
    def orientationInterpolator(set_fraction, key, keyValue):
        """Interpola entre uma lista de valores de rotação especificos."""
        # https://www.web3d.org/specifications/X3Dv4/ISO-IEC19775-1v4-IS/Part01/components/interpolators.html#OrientationInterpolator
        # Interpola rotações são absolutas no espaço do objeto e, portanto, não são cumulativas.
        # Uma orientação representa a posição final de um objeto após a aplicação de uma rotação.
        # Um OrientationInterpolator interpola entre duas orientações calculando o caminho mais
        # curto na esfera unitária entre as duas orientações. A interpolação é linear em
        # comprimento de arco ao longo deste caminho. Os resultados são indefinidos se as duas
        # orientações forem diagonalmente opostas. O campo keyValue possui uma lista com os
        # valores a serem interpolados, key possui uma lista respectiva de chaves
        # dos valores em keyValue, a fração a ser interpolada vem de set_fraction que varia de
        # zeroa a um. O campo keyValue deve conter exatamente tantas rotações 3D quanto os
        # quadros-chave no key.

        # O print abaixo é só para vocês verificarem o funcionamento, DEVE SER REMOVIDO.
        print("OrientationInterpolator : set_fraction = {0}".format(set_fraction))
        print("OrientationInterpolator : key = {0}".format(key)) # imprime no terminal
        print("OrientationInterpolator : keyValue = {0}".format(keyValue))

        # Abaixo está só um exemplo de como os dados podem ser calculados e transferidos
        value_changed = [0, 0, 1, 0]

        return value_changed

    # Para o futuro (Não para versão atual do projeto.)
    def vertex_shader(self, shader):
        """Para no futuro implementar um vertex shader."""

    def fragment_shader(self, shader):
        """Para no futuro implementar um fragment shader."""

    # --------------------------------------------------------------- #