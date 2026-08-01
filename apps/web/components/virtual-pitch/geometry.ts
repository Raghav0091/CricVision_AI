import * as THREE from "three";

import {
  toThreeVector,
  type CricketVector3,
  type ThreeVector3
} from "@/lib/virtual-pitch";

import type { ThreeCoordinate } from "./rendererTypes";


const UP = new THREE.Vector3(0, 1, 0);
const RIGHT = new THREE.Vector3(1, 0, 0);


export function asThreeVector(point: CricketVector3 | ThreeCoordinate): THREE.Vector3 {
  if (Array.isArray(point)) {
    return new THREE.Vector3(point[0], point[1], point[2]);
  }
  const vector = toThreeVector(point as CricketVector3);
  return new THREE.Vector3(vector.x, vector.y, vector.z);
}


export function sceneVector(point: ThreeVector3): THREE.Vector3 {
  return new THREE.Vector3(point.x, point.y, point.z);
}


export function cylinderTransform(start: THREE.Vector3, end: THREE.Vector3) {
  const direction = end.clone().sub(start);
  const length = direction.length();
  const quaternion = new THREE.Quaternion();
  if (length > Number.EPSILON) {
    quaternion.setFromUnitVectors(UP, direction.normalize());
  }
  return {
    length,
    midpoint: start.clone().add(end).multiplyScalar(0.5),
    quaternion
  };
}


export function stripTransform(start: THREE.Vector3, end: THREE.Vector3) {
  const direction = end.clone().sub(start);
  const length = direction.length();
  const quaternion = new THREE.Quaternion();
  if (length > Number.EPSILON) {
    quaternion.setFromUnitVectors(RIGHT, direction.normalize());
  }
  return {
    length,
    midpoint: start.clone().add(end).multiplyScalar(0.5),
    quaternion
  };
}


export function polygonGeometry(vertices: readonly CricketVector3[]): THREE.BufferGeometry {
  const points = vertices.map(asThreeVector);
  const projected = vertices.map((point) => new THREE.Vector2(point.x, point.y));
  const triangles = THREE.ShapeUtils.triangulateShape(projected, []);
  const positions = new Float32Array(triangles.length * 9);
  triangles.forEach((triangle, triangleIndex) => {
    triangle.forEach((vertexIndex, cornerIndex) => {
      const point = points[vertexIndex];
      const offset = triangleIndex * 9 + cornerIndex * 3;
      positions[offset] = point.x;
      positions[offset + 1] = point.y;
      positions[offset + 2] = point.z;
    });
  });
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}
