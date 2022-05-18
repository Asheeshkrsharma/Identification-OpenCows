import * as THREE from 'three';
import { RenderContext } from './render';
export interface ScatterPlotVisualizer {
    id: string;
    setScene(scene: THREE.Scene): void;
    dispose(): void;
    onPointPositionsChanged(newWorldSpacePointPositions: Float32Array): void;
    onPickingRender(renderContext: RenderContext): void;
    onRender(renderContext: RenderContext): void;
    onResize(newWidth: number, newHeight: number): void;
}
