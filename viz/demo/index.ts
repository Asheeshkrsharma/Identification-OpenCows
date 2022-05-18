/*
@license
Copyright 2019 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

import { data } from './data/projection';
import { Points, Dataset, PointMetadata } from '../src/data';
import { UserStyles, PointStyles } from '../src/styles';
import * as math from 'mathjs';
import * as util from '../src/util';
// @ts-ignore
import jStat = require('jStat');
// @ts-ignore
import numeric = require('numeric');
import * as THREE from 'three';

// import { makeSequences } from './sequences';
import { ScatterGL, RenderMode } from '../src';
import { SCATTER_PLOT_CUBE_LENGTH } from '../src/constants';
// @ts-ignore
import {LineMaterial} from './lineMaterial';
// @ts-ignore
import { LineGeometry } from './LineGeometry';
// @ts-ignore
import { Line2 } from './Line2';

/** SAFEHTML */

const dataPoints: Points = [];
const metadata: PointMetadata[] = [];
data.projection.forEach((vector, index: number) => {
  const labelIndex = data.labels[index];
  dataPoints.push(vector);
  metadata.push({
    labelIndex,
    label: data.labelNames[labelIndex],
    index: index
  });
});

// const sequences = makeSequences(dataPoints, metadata);
const dataset = new Dataset(dataPoints, metadata);

dataset.setSpriteMetadata({
  spriteImage: 'spritesheet.png',
  singleSpriteSize: [28, 20],
  // Uncomment the following line to only use the first sprite for every point
  // spriteIndices: dataPoints.map(d => 0),
});

let lastSelectedPoints: number[] = [];
let renderMode = 'points';

const containerElement = document.getElementById('container')!;
const messagesElement = document.getElementById('messages')!;

const setMessage = (message: string) => {
  const messageStr = `${message}`;
  messagesElement.innerHTML = messageStr;
};

const scatterGL = new ScatterGL(containerElement, {
  onClick: (point: number | null) => {
    const index = point ?? 0
    if (index !== 0 && point !== null) {
      let message = `Image index: ${point}, Path:${data.path![index] ?? ''} ${index}`;
      setMessage(message);
    }
    else {
      setMessage('');
    }
  },

  onHover: (point: number | null) => {
    const index = point ?? 0
    if (index !== 0 && point !== null) {
      let message = `Image index: ${point}, Path:${data.path![index] ?? ''} ${index}`;
      setMessage(message);
    }
    else {
      setMessage('');
    }
  },
  renderMode: RenderMode.POINT,
  orbitControls: {
    zoomSpeed: 1.125,
  },
  styles: {
    point: {
      scaleDefault: 3,
      // colorUnselected: Color
      colorNoSelection: `rgba(66, 133, 244, 1)`,
      // colorSelected: Color;
      colorHover: '#4285F4',
      scaleSelected: 6,
      scaleHover: 4
    },
    fog: {
      enabled: true,
      threshold: 0.5,
    },
    sprites: {
      minPointSize: 15.0,
      // imageSize: 10,
      colorUnselected: '#4285F4',
      colorNoSelection: '#ffffff',
    },
    label: {
      fontSize: 8,
      scaleDefault: 6,
      fillColorSelected: '#000000',
      fillColorHover: '#4285F4',
      strokeColorSelected: '#ffffff',
      strokeColorHover: '#ffffff',
      strokeWidth: 6,
      fillWidth: 3,
    },
  },
});
scatterGL.setParameters({
  onSelect: (points: number[]) => {
    let message = '';
    if (points.length === 0 && lastSelectedPoints.length === 0) {
      message = 'no selection';
      // Delete previous lie if any
      let previous = scatterGL.pointVisualizer?.scene.getObjectByName('line') as THREE.Object3D;
      scatterGL.pointVisualizer?.scene.remove(previous);
      
    } else if (points.length === 0 && lastSelectedPoints.length > 0) {
      message = 'deselected';
    } else if (points.length === 1) {
      message = `selected ${points}`;
    } else {
      // https://www.xarg.org/2018/04/how-to-plot-a-covariance-error-ellipse/
      let plotErrorEllipse = function(mu: number[], Sigma: number[][], p: number, dev: number[]) {

        p = p || 0.95;

        var s = -2 * Math.log(1 - p);

        var a = Sigma[0][0];
        var b = Sigma[0][1];
        var c = Sigma[1][0];
        var d = Sigma[1][1];

        var tmp = Math.sqrt((a - d) * (a - d) + 4 * b * c);
        var V = [
          [-(tmp - a + d) / (2 * c), (tmp + a - d) / (2 * c)],
          [1, 1]
        ];
        var sqrtD = [
          Math.sqrt(s * (a + d - tmp) / 2),
          Math.sqrt(s * (a + d + tmp) / 2)
        ];

        var norm1 = Math.hypot(V[0][0], 1);
        var norm2 = Math.hypot(V[0][1], 1);
        V[0][0] /= norm1;
        V[1][0] /= norm1;
        V[0][1] /= norm2;
        V[1][1] /= norm2;

        var ndx = sqrtD[0] < sqrtD[1] ? 1 : 0;

        var x1 = mu[0] + V[0][ndx] * sqrtD[ndx];
        var y1 = mu[1] + V[1][ndx] * sqrtD[ndx];

        var x2 = mu[0] + V[0][1 - ndx] * sqrtD[1 - ndx];
        var y2 = mu[1] + V[1][1 - ndx] * sqrtD[1 - ndx];
        let theta = Math.atan2(y1 - mu[1], x1 - mu[0]);
        if (theta < 0) {
          theta += 2 * Math.PI;
        }
        
        const ellipse = new THREE.EllipseCurve(
          mu[0], mu[1],
          Math.hypot(x1 - mu[0], y1 - mu[1]),
          Math.hypot(x2 - mu[0], y2 - mu[1]), 0, Math.PI * 2, false, theta);
        return ellipse
      }

      const level = 0.9;
      let Xs = dataset.points.map(p => p[0]);
      let Ys = dataset.points.map(p => p[1]);
      let xExtent = util.extent(Xs);
      let yExtent = util.extent(Ys);
      const halfCube = SCATTER_PLOT_CUBE_LENGTH / 2;
      const getRange = (extent: number[]) => Math.abs(extent[1] - extent[0]);
      const xRange = getRange(xExtent);
      const yRange = getRange(yExtent);
      const maxRange = Math.max(getRange(xExtent), getRange(yExtent), getRange([0, 0]));
      const makeScaleRange = (range: number, base: number) => [
        -base * (range / maxRange),
        base * (range / maxRange),
      ];
      const xScale = makeScaleRange(xRange, halfCube);
      const yScale = makeScaleRange(yRange, halfCube);
      Xs = points.map(p => util.scaleLinear(dataPoints[p][0], xExtent, xScale));
      Ys = points.map(p => util.scaleLinear(dataPoints[p][1], yExtent, yScale));

      let xDataDev = math.std(Xs) as unknown as number;
      let xMean = math.mean(Xs);
      let yDataDev = math.std(Ys) as unknown as number;
      let yMean = math.mean(Ys);

      let cor = jStat.corrcoeff(Xs, Ys);
      let cov = cor * xDataDev * yDataDev

      let covmat = [
        [math.pow(xDataDev, 2), cov],
        [cov, math.pow(yDataDev, 2)]
      ] as unknown as number[][];

      const ellipse = plotErrorEllipse([xMean, yMean], covmat, level, [xDataDev, yDataDev]);

      // Delete previous lie if any
      let previous = scatterGL.pointVisualizer?.scene.getObjectByName('line') as THREE.Object3D;
      scatterGL.pointVisualizer?.scene.remove(previous);

      // Draw the ellipse
      const material = new LineMaterial( {
        color: 0xBCBCBC,
        linewidth: 0.001, // in world units with size attenuation, pixels otherwise
        vertexColors: true,
        //resolution:  // to be set by renderer, eventually
        dashed: false,
        alphaToCoverage: false,
        dashScale: 50
      } );

      const numberPoints = 6000;
      var ellipseGeometry = new THREE.BufferGeometry().setFromPoints(ellipse.getPoints(numberPoints));

      // Compute colors
      const divisions = Math.round( 12 * numberPoints);
      const color = new THREE.Color();
      const colors = [];
      for ( let i = 0, l = divisions; i < l; i ++ ) {
        color.setHex(0xBCBCBC);
        colors.push( color.r, color.g,  color.b );
      }
      let positions = ellipseGeometry.getAttribute('position')['array'];
      var lineGeometry = new LineGeometry().setPositions(positions);
      lineGeometry.setColors( colors );
      let line = new Line2(lineGeometry, material);
      line.computeLineDistances();
      line.scale.set( 1, 1, 1 );
      line.name = 'line';
      line.frustumCulled = false;
      scatterGL.pointVisualizer?.scene.add(line)
      message = `selected ${points.length} points`;
    }
    setMessage(message);
  },
})

// const colors = [[170, 56, 57], [102, 229, 51], [172, 59, 227], [163, 241, 57], [120, 76, 230], [210, 242, 57], [55, 81, 219], [233, 231, 44], [146, 74, 211], [94, 233, 92], [226, 72, 220], [133, 218, 67], [172, 52, 183], [64, 177, 44], [192, 108, 238],
// [181, 228, 82], [90, 112, 241], [236, 209, 51], [104, 88, 208], [136, 187, 34], [60, 77, 187], [202, 213, 67], [126, 71, 176], [109, 210, 98], [219, 45, 168], [40, 199, 105], [254, 29, 102], [106, 239, 144], [247, 60, 130], [81, 240, 177],
// [240, 47, 20], [93, 239, 208], [229, 57, 39], [65, 205, 219], [233, 83, 20], [90, 141, 238], [237, 146, 23], [69, 104, 204], [234, 247, 117], [92, 81, 177], [142, 177, 43], [152, 125, 238], [54, 143, 35], [232, 98, 196], [174, 235, 125],
// [170, 61, 148], [94, 174, 77], [214, 125, 227], [183, 179, 46], [43, 106, 190], [230, 182, 54], [73, 75, 147], [234, 217, 109], [107, 79, 160], [148, 198, 96], [147, 81, 168], [113, 158, 58], [221, 62, 137], [85, 189, 121], [215, 31, 99],
// [78, 203, 170], [232, 59, 61], [118, 232, 227], [226, 54, 80], [41, 147, 90], [238, 116, 179], [68, 119, 29], [223, 163, 240], [150, 160, 46], [126, 114, 200], [201, 227, 133], [93, 92, 167], [184, 143, 41], [60, 147, 221], [195, 78, 26],
// [64, 184, 225], [183, 40, 33], [161, 235, 169], [171, 49, 108], [75, 143, 69], [197, 112, 187], [49, 114, 48], [161, 146, 232], [125, 132, 31], [53, 87, 153], [223, 141, 55], [35, 110, 173], [240, 120, 66], [108, 177, 240], [157, 70, 25],
// [109, 172, 219], [228, 96, 80], [44, 165, 135], [180, 45, 79], [139, 213, 182], [223, 87, 110], [45, 129, 97], [203, 103, 144], [68, 90, 6], [184, 178, 240], [132, 116, 21], [89, 117, 187], [212, 176, 91], [68, 80, 130], [227, 238, 166],
// [120, 90, 153], [177, 179, 91], [68, 80, 130], [213, 203, 135], [68, 80, 130], [161, 207, 143], [140, 70, 117], [131, 174, 110], [169, 132, 195], [94, 114, 34], [132, 148, 213], [154, 98, 29], [55, 125, 169], [193, 115, 55], [93, 103, 156],
// [245, 197, 140], [26, 100, 71], [238, 132, 157], [35, 94, 49], [233, 162, 211], [84, 86, 8], [164, 101, 147], [61, 119, 72], [229, 128, 111], [49, 156, 154], [199, 100, 71], [24, 120, 106], [245, 165, 135], [65, 90, 31], [145, 67, 87], [94, 158, 117],
// [180, 93, 95], [89, 114, 54], [220, 140, 137], [104, 104, 29], [148, 72, 51], [137, 151, 81], [129, 74, 40], [210, 169, 115], [82, 86, 32], [224, 149, 101], [99, 81, 10], [178, 141, 78], [112, 75, 12], [134, 126, 73], [152, 105, 65], [138, 126, 54],
// [139, 99, 37], [104, 100, 48], [118, 99, 30]];

const colors = [
  [102, 162, 209], [222, 74, 41], [132, 96, 233], [94, 181, 49],
  [102, 54, 184], [194, 180, 29], [78, 31, 140], [72, 182, 92],
  [198, 92, 220], [151, 171, 52], [130, 123, 235], [221, 139, 36],
  [76, 91, 188], [159, 146, 46], [140, 68, 172], [83, 141, 65],
  [50, 21, 99], [61, 173, 125], [214, 54, 79], [70, 183, 195],
  [175, 66, 40], [83, 138, 229], [195, 128, 52], [84, 62, 134],
  [137, 135, 60], [191, 127, 216], [97, 147, 110], [208, 72, 114],
  [73, 148, 134], [224, 124, 94], [107, 154, 223], [114, 50, 30],
  [92, 104, 176], [179, 126, 75], [48, 28, 72], [144, 146, 93],
  [173, 140, 208], [71, 76, 39], [162, 120, 170], [65, 141, 150],
  [203, 105, 106], [86, 98, 151], [188, 133, 126], [59, 58, 93],
  [137, 159, 193], [101, 40, 52], [72, 99, 130],
]
const heavyTransparentColorsByLabel = colors.map(rgb => `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.75)`);

window.addEventListener('load', () => {
  document
    .querySelectorAll<HTMLInputElement>('input[name="interactions"]')
    .forEach(inputElement => {
      inputElement.addEventListener('change', () => {
        if (inputElement.value === 'pan') {
          scatterGL.setPanMode();
        } else if (inputElement.value === 'select') {
          scatterGL.setSelectMode();
        }
      });
    });

  document
    .querySelectorAll<HTMLInputElement>('input[name="render"]')
    .forEach(inputElement => {
      inputElement.addEventListener('change', () => {
        renderMode = inputElement.value;
        if (inputElement.value === 'points') {
          scatterGL.setPointRenderMode();
        } else if (inputElement.value === 'sprites') {
          scatterGL.setSpriteRenderMode();
        }
      });
    });

  function updateDefault() {
    scatterGL.setPointColorer((i, selectedIndices, hoverIndex) => {
      const pointIndex = dataset.metadata![i]['index'] as number;
      // Use the label index to decide whether is is the the last element
      // of the array
      let isLast = ((pointIndex + 1) % 5) == 0;
      let newCol = selectedIndices.size === 0 ? `rgba(0, 0, 0, 0.5)` : 
      (selectedIndices.has(i) ? `rgba(0, 0, 0, 0.5)` : `rgba(188, 188, 188, 0.5)`);
      
      if (isLast) {
        newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,1)');
      } else {
        newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,0.1)');
      }
      return newCol
    });
  }

  document
    .querySelectorAll<HTMLInputElement>('input[name="color"]')
    .forEach(inputElement => {
      inputElement.addEventListener('change', () => {
        if (inputElement.value === 'default') {
          updateDefault();
        } else if (inputElement.value === 'label') {
          scatterGL.setPointColorer((i, selectedIndices, hoverIndex) => {
            const labelIndex = dataset.metadata![i]['labelIndex'] as number;
            const pointIndex = dataset.metadata![i]['index'] as number;

            let isLast = ((pointIndex + 1) % 5) == 0;
            // If nothing is selected, return the heavy color Otherwise, keep the selected points heavy and non-selected white
            var newCol = selectedIndices.size === 0 ? heavyTransparentColorsByLabel[labelIndex] :
              (selectedIndices.has(i) ? heavyTransparentColorsByLabel[labelIndex] : `rgba(188, 188, 188, 0.5)`);

            // Use the label index to decide whether is is the the last element
            // of the array
            if (isLast) {
              newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,1)');
            } else {
              newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,0.1)');
            }
            return newCol;
          });
        }
      });
    });
  scatterGL.setPointRenderMode();
  scatterGL.render(dataset);
  updateDefault();
});

// Add in a resize observer for automatic window resize.
window.addEventListener('resize', () => {
  scatterGL.resize();
});