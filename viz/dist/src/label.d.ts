export interface BoundingBox {
    loX: number;
    loY: number;
    hiX: number;
    hiY: number;
}
export declare class CollisionGrid {
    private numHorizCells;
    private numVertCells;
    private grid;
    private bound;
    private cellWidth;
    private cellHeight;
    constructor(bound: BoundingBox, cellWidth: number, cellHeight: number);
    private boundWidth;
    private boundHeight;
    private boundsIntersect;
    insert(bound: BoundingBox, justTest?: boolean): boolean;
    private getCellX;
    private getCellY;
}
