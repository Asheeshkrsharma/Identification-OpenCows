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
// import { makeSequences } from './sequences';
import { ScatterGL, RenderMode } from '../src';
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
  const messageStr = `🔥 ${message}`;
  // console.log(messageStr);
  messagesElement.innerHTML = messageStr;
};

const scatterGL = new ScatterGL(containerElement, {
  onClick: (point: number | null) => {
    const index = point ?? 0
    if (index !== 0 && point !== null) {
      let message = `hover ${point} ${data.path![index] ?? ''} ${index}`;
      setMessage(message);
    }
    else {
      setMessage('');
    }
  },

  onHover: (point: number | null) => {
    const index = point ?? 0
    if (index !== 0 && point !== null) {
      let message = `hover ${point} ${data.path![index] ?? ''}  ${index}`;
      setMessage(message);
    }
    else {
      setMessage('');
    }
  },
  onSelect: (points: number[]) => {
    let message = '';
    if (points.length === 0 && lastSelectedPoints.length === 0) {
      message = 'no selection';
    } else if (points.length === 0 && lastSelectedPoints.length > 0) {
      message = 'deselected';
    } else if (points.length === 1) {
      message = `selected ${points}`;
    } else {
      message = `selected ${points.length} points`;
    }
    setMessage(message);
  },
  renderMode: RenderMode.POINT,
  orbitControls: {
    zoomSpeed: 1.125,
  },
  styles: {
    point: {
      scaleDefault: 2
    },
    fog: {
      enabled: true,
      threshold: 0.5,
    },
    sprites: {
      minPointSize: 5.0,
      imageSize: 28,
      colorUnselected: '#ffffff',
      colorNoSelection: '#ffffff',
    },
    label: {
      fontSize: 8,
      scaleDefault: 5,
      fillColorSelected: '#000000',
      fillColorHover: '#000000',
      strokeColorSelected: '#ffffff',
      strokeColorHover: '#ffffff',
      strokeWidth: 5,
      fillWidth: 3,
    },
  },
});

scatterGL.setPointRenderMode();
scatterGL.render(dataset);

// Add in a resize observer for automatic window resize.
window.addEventListener('resize', () => {
  scatterGL.resize();
});

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
      } else if (inputElement.value === 'text') {
        scatterGL.setTextRenderMode();
      }
    });
  });


const colors = [[170, 56, 57], [102, 229, 51], [172, 59, 227], [163, 241, 57], [120, 76, 230], [210, 242, 57], [55, 81, 219], [233, 231, 44], [146, 74, 211], [94, 233, 92], [226, 72, 220], [133, 218, 67], [172, 52, 183], [64, 177, 44], [192, 108, 238],
[181, 228, 82], [90, 112, 241], [236, 209, 51], [104, 88, 208], [136, 187, 34], [60, 77, 187], [202, 213, 67], [126, 71, 176], [109, 210, 98], [219, 45, 168], [40, 199, 105], [254, 29, 102], [106, 239, 144], [247, 60, 130], [81, 240, 177],
[240, 47, 20], [93, 239, 208], [229, 57, 39], [65, 205, 219], [233, 83, 20], [90, 141, 238], [237, 146, 23], [69, 104, 204], [234, 247, 117], [92, 81, 177], [142, 177, 43], [152, 125, 238], [54, 143, 35], [232, 98, 196], [174, 235, 125],
[170, 61, 148], [94, 174, 77], [214, 125, 227], [183, 179, 46], [43, 106, 190], [230, 182, 54], [73, 75, 147], [234, 217, 109], [107, 79, 160], [148, 198, 96], [147, 81, 168], [113, 158, 58], [221, 62, 137], [85, 189, 121], [215, 31, 99],
[78, 203, 170], [232, 59, 61], [118, 232, 227], [226, 54, 80], [41, 147, 90], [238, 116, 179], [68, 119, 29], [223, 163, 240], [150, 160, 46], [126, 114, 200], [201, 227, 133], [93, 92, 167], [184, 143, 41], [60, 147, 221], [195, 78, 26],
[64, 184, 225], [183, 40, 33], [161, 235, 169], [171, 49, 108], [75, 143, 69], [197, 112, 187], [49, 114, 48], [161, 146, 232], [125, 132, 31], [53, 87, 153], [223, 141, 55], [35, 110, 173], [240, 120, 66], [108, 177, 240], [157, 70, 25],
[109, 172, 219], [228, 96, 80], [44, 165, 135], [180, 45, 79], [139, 213, 182], [223, 87, 110], [45, 129, 97], [203, 103, 144], [68, 90, 6], [184, 178, 240], [132, 116, 21], [89, 117, 187], [212, 176, 91], [68, 80, 130], [227, 238, 166],
[120, 90, 153], [177, 179, 91], [68, 80, 130], [213, 203, 135], [68, 80, 130], [161, 207, 143], [140, 70, 117], [131, 174, 110], [169, 132, 195], [94, 114, 34], [132, 148, 213], [154, 98, 29], [55, 125, 169], [193, 115, 55], [93, 103, 156],
[245, 197, 140], [26, 100, 71], [238, 132, 157], [35, 94, 49], [233, 162, 211], [84, 86, 8], [164, 101, 147], [61, 119, 72], [229, 128, 111], [49, 156, 154], [199, 100, 71], [24, 120, 106], [245, 165, 135], [65, 90, 31], [145, 67, 87], [94, 158, 117],
[180, 93, 95], [89, 114, 54], [220, 140, 137], [104, 104, 29], [148, 72, 51], [137, 151, 81], [129, 74, 40], [210, 169, 115], [82, 86, 32], [224, 149, 101], [99, 81, 10], [178, 141, 78], [112, 75, 12], [134, 126, 73], [152, 105, 65], [138, 126, 54],
[139, 99, 37], [104, 100, 48], [118, 99, 30]];

const hues = [...new Array(155)].map((_, i) => Math.floor((255 / 155) * i));

const lightTransparentColorsByLabel = colors.map(rgb => `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.05)`);
const heavyTransparentColorsByLabel = colors.map(rgb => `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 0.75)`);
const opaqueColorsByLabel = colors.map(rgb => `rgba(${rgb[0]}, ${rgb[1]}, ${rgb[2]}, 1.)`);

// const lightTransparentColorsByLabel = hues.map(
//   hue => `hsla(${hue}, 100%, 50%, 0.05)`
// );
// const heavyTransparentColorsByLabel = hues.map(
//   hue => `hsla(${hue}, 100%, 50%, 0.75)`
// );
// const opaqueColorsByLabel = hues.map(hue => `hsla(${hue}, 100%, 50%, 1)`);

document
  .querySelectorAll<HTMLInputElement>('input[name="color"]')
  .forEach(inputElement => {
    inputElement.addEventListener('change', () => {
      if (inputElement.value === 'default') {
        scatterGL.setPointColorer(null);
      } else if (inputElement.value === 'label') {
        scatterGL.setPointColorer((i, selectedIndices, hoverIndex) => {
          const labelIndex = dataset.metadata![i]['labelIndex'] as number;
          const pointIndex = dataset.metadata![i]['index'] as number;
          // Use the label index to decide whether is is the the last element
          // of the array
          let isLast = ((pointIndex + 1) % 5) == 0;
          const opaque = renderMode !== 'points';
          if (opaque) {
            let newCol = opaqueColorsByLabel[labelIndex];
            if (isLast) {
              newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,1)');
            } else {
              newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,0.1)');
            }
            return newCol;
          } else {
            if (hoverIndex === i) {
              return 'red';
            }

            // If nothing is selected, return the heavy color
            if (selectedIndices.size === 0) {
              let newCol = heavyTransparentColorsByLabel[labelIndex];
              if (isLast) {
                newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,1)');
              } else {
                newCol = newCol.replace(/rgba?(\(\s*\d+\s*,\s*\d+\s*,\s*\d+)(?:\s*,.+?)?\)/, 'rgba$1,0.1)');
              }
              return newCol;
            }

            // Otherwise, keep the selected points heavy and non-selected light
            else {
              const isSelected = selectedIndices.has(i);
              return isSelected
                ? heavyTransparentColorsByLabel[labelIndex]
                : lightTransparentColorsByLabel[labelIndex];
            }
          }
        });
      }
    });
  });

const dimensionsToggle = document.querySelector<HTMLInputElement>(
  'input[name="3D"]'
)!;
dimensionsToggle.addEventListener('change', (e: any) => {
  const is3D = dimensionsToggle.checked;
  scatterGL.setDimensions(is3D ? 3 : 2);
});

// const sequencesToggle = document.querySelector<HTMLInputElement>(
//   'input[name="sequences"]'
// )!;
// sequencesToggle.addEventListener('change', (e: any) => {
//   const showSequences = sequencesToggle.checked;
//   scatterGL.setSequences(showSequences ? sequences : []);
// });

// Set up controls for buttons
const selectRandomButton = document.getElementById('select-random')!;
selectRandomButton.addEventListener('click', () => {
  const randomIndex = Math.floor(dataPoints.length * Math.random());
  scatterGL.select([randomIndex]);
});

const toggleOrbitButton = document.getElementById('toggle-orbit')!;
toggleOrbitButton.addEventListener('click', () => {
  if (scatterGL.isOrbiting()) {
    scatterGL.stopOrbitAnimation();
  } else {
    scatterGL.startOrbitAnimation();
  }
});
