import * as THREE from 'three';
import { ScatterPlotVisualizer } from './scatter_plot_visualizer';
import { RenderContext } from './render';
import { Styles } from './styles';
export interface SpriteSheetParams {
    spritesheetImage: HTMLImageElement | string;
    spriteDimensions: [number, number];
    spriteIndices: Float32Array;
    onImageLoad: () => void;
}
export declare class ScatterPlotVisualizerSprites implements ScatterPlotVisualizer {
    private styles;
    id: string;
    private scene;
    private fog;
    private texture;
    private standinTextureForPoints;
    private spriteIndexBufferAttribute;
    private renderMaterial;
    private pickingMaterial;
    private isSpriteSheetMode;
    private spriteSheetParams;
    private spriteSheetImage;
    private spritesPerRow;
    private spritesPerColumn;
    private spriteDimensions;
    private points;
    private worldSpacePointPositions;
    private pickingColors;
    private renderColors;
    constructor(styles: Styles, spriteSheetParams?: SpriteSheetParams);
    private createUniforms;
    private createRenderMaterial;
    private createPickingMaterial;
    private createPointSprites;
    private calculatePointSize;
    private createGeometry;
    private setFogDistances;
    dispose(): void;
    private disposeGeometry;
    private disposeSpriteSheet;
    setScene(scene: THREE.Scene): void;
    private setSpriteSheet;
    private setSpriteIndexBuffer;
    onPointPositionsChanged(newPositions: Float32Array): void;
    onPickingRender(rc: RenderContext): void;
    onRender(rc: RenderContext): void;
    onResize(newWidth: number, newHeight: number): void;
}
