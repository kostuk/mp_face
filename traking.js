import "./style.css";
import vision from "https://cdn.skypack.dev/@mediapipe/tasks-vision@0.1.0-alpha-11";
const { FaceLandmarker, FilesetResolver } = vision;
// import { FaceLandmarker, FilesetResolver } from "@mediapipe/tasks-vision";
import { Camera } from "@mediapipe/camera_utils";
import { drawConnectors } from "@mediapipe/drawing_utils";
import Plotly from 'plotly.js-dist';
import {
  FaceMesh,
  // индексы координат (см. ниже)
  FACEMESH_FACE_OVAL,
  FACEMESH_LEFT_EYE,
  FACEMESH_LEFT_EYEBROW,
  FACEMESH_LEFT_IRIS,
  FACEMESH_LIPS,
  FACEMESH_RIGHT_EYE,
  FACEMESH_RIGHT_EYEBROW,
  FACEMESH_RIGHT_IRIS,
  FACEMESH_TESSELATION,
} from "@mediapipe/face_mesh";
let runningMode= 'IMAGE';//'IMAGE' | 'VIDEO'
let webcamRunning = false;
const videoWidth = 480;

const enableWebcamButton =  document.getElementById('webcamButton') ;
const videoBlendShapes = document.getElementById('video-blend-shapes');
const videoAtShapes = document.getElementById('video-at-shapes');

// Our input frames will come from here.
const video = document.getElementById('webcam') ;
const canvasElement = document.getElementById('output_canvas') ;

const canvasCtx = canvasElement.getContext('2d');
const  mouthOpen = {'mouthPucker':-0.6};

const squint = {'eyeSquintLeft':0.7,'eyeLookInLeft':0.7, 'eyeLookOutLeft':0.7,'eyeSquintRight':0.7,'eyeLookInRight':0.7, 'eyeLookOutRight':0.7,'browDownLeft':0.4,'browDownRight':0.4};
let faceLandmarker;
let attentionTime = [];
let lastAttentionTime={time:null, p:0, m:0,e:0, c:0};

// Before we can use HandLandmarker class we must wait for it to finish
// loading. Machine Learning models can be large and take a moment to
// get everything needed to run.
async function runDemo() {
  // Read more `CopyWebpackPlugin`, copy wasm set from "https://cdn.skypack.dev/node_modules" to `/wasm`
  const filesetResolver = await FilesetResolver.forVisionTasks('https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.1.0-alpha-11/wasm');
  faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
    baseOptions: {
      modelAssetPath: `https://storage.googleapis.com/mediapipe-assets/face_landmarker_with_blendshapes.task?generation=1678504998301299`,
    },
    outputFaceBlendshapes: true,
    runningMode,
    numFaces: 1
  });
}
runDemo();

// Check if webcam access is supported.
function hasGetUserMedia() {
  return !!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia);
}

// If webcam supported, add event listener to button for when user
// wants to activate it.7
if (hasGetUserMedia()) {
  enableWebcamButton.addEventListener('click', enableCam);
} else {
  console.warn('getUserMedia() is not supported by your browser');
}
// Enable the live webcam view and start detection.
function enableCam(event) {
  if (!faceLandmarker) {
    console.log('Wait! faceLandmarker not loaded yet.');
    return;
  }

  if (webcamRunning === true) {
    webcamRunning = false;
    enableWebcamButton.innerText = 'ENABLE PREDICTIONS';
  } else {
    console.log('webcam was off');
    webcamRunning = true;
    enableWebcamButton.innerText = 'DISABLE PREDICITONS';
    initChart();
  }

  // getUsermedia parameters.
  const constraints = {
    video: true
  };

  // Activate the webcam stream.
  navigator.mediaDevices.getUserMedia(constraints).then(function (stream) {
    video.srcObject = stream;
    video.addEventListener('loadeddata', predictWebcam);
  });
}

async function predictWebcam() {
  const radio = (video.videoHeight/video.videoWidth)
  video.style.width = videoWidth + "px";
  video.style.height = (videoWidth*radio) + "px" ;
  canvasElement.style.width = videoWidth + "px" ;
  canvasElement.style.height = (videoWidth*radio) + "px" ;
  canvasElement.width = video.videoWidth;
  canvasElement.height = video.videoHeight;
  // Now let's start detecting the stream.
  if (runningMode === 'IMAGE') {
    runningMode = 'VIDEO';
    await faceLandmarker.setOptions({ runningMode: runningMode });
  }
  let nowInMs = Date.now();
  const results = faceLandmarker.detectForVideo(video, nowInMs);

  canvasCtx.save();
  canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
  
  if (results.faceLandmarks) {
    for (const landmarks of results.faceLandmarks) {
//      drawConnectors(canvasCtx, landmarks, FACEMESH_TESSELATION,{color: '#C0C0C070', lineWidth: 1});
      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYE, {color: '#30FF30'});
//      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_EYEBROW, {color: '#FF3030', lineWidth: 1});
      drawConnectors(canvasCtx, landmarks, FACEMESH_RIGHT_IRIS, {color: '#FF3030'});
      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYE, {color: '#30FF30'});
//      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_EYEBROW, {color: '#FF3030', lineWidth: 1});
      drawConnectors(canvasCtx, landmarks, FACEMESH_LEFT_IRIS, {color: '#FF3030'});
      drawConnectors(canvasCtx, landmarks, FACEMESH_FACE_OVAL, {color: '#E0E0E0', lineWidth: 1});
      drawConnectors(canvasCtx, landmarks, FACEMESH_LIPS, {color: '#E0E0E0'});
    }
  }
  canvasCtx.restore();
//  console.log(results)
  if(!results.faceLandmarks || results.faceBlendshapes.length==0){
    lastAttentionTime.p+=0;
    lastAttentionTime.e+=0;
    lastAttentionTime.m+=0;
    lastAttentionTime.c++;
    drawBlendShapesExt(videoAtShapes,{p:0,e:0,m:0});
  }else{
    let attentionTime = {p:0, e:0,m:0};
    attentionTime.p=1;
    let blendShapes = results.faceBlendshapes[0];
    let l=0.0001;
    for (const name in mouthOpen) {
      let v =0;
      let h = mouthOpen[name]
      blendShapes.categories.map((shape) => {
        if((shape.displayName || shape.categoryName)==name){
            let val=0;
            if(h>0 && shape.score>h){
              val= (shape.score-h)/(1-h)
            }else if(h<0 && shape.score<-h){
              val = (-h-shape.score)/(-h);
            }
            if(val<0 || val>1)
              debugger;
            if(val>0){
                v +=val;
                l++;
              }
        }
      });
      attentionTime.m = v/l;
    }
    l=0.0001;
    for (const name in squint) {
      let v =0;
      let h = squint[name]
      blendShapes.categories.map((shape) => {
        if((shape.displayName || shape.categoryName)==name){
          let val=0;
          if(h>0 && shape.score>h){
            val= (shape.score-h)/(1-h)
          }else if(h<0 && shape.score<-h){
            val = (-h-shape.score)/(-h);
          }
          if(val<0 || val>1)
            debugger;
          if(val>0){
              v +=val;
              l++;
            }

        }
      });
      attentionTime.e+= v/l;
    }
    drawBlendShapesExt(videoAtShapes, attentionTime);
    
    lastAttentionTime.p+=attentionTime.p;
    lastAttentionTime.e+=attentionTime.e;
    lastAttentionTime.m+=attentionTime.m;
    lastAttentionTime.c++;

  }
  if(lastAttentionTime.time+1000< Date.now()){
    lastAttentionTime.p=lastAttentionTime.p/lastAttentionTime.c;
    lastAttentionTime.e=lastAttentionTime.p*(1-lastAttentionTime.e/lastAttentionTime.c);
    lastAttentionTime.m=lastAttentionTime.p*(1-lastAttentionTime.m/lastAttentionTime.c);
    attentionTime.push(lastAttentionTime);
    drawChart(lastAttentionTime)
    lastAttentionTime={time:Date.now(), p:0, m:0,e:0, c:0};
  }
  drawBlendShapes(videoBlendShapes, results.faceBlendshapes);

  // Call this function again to keep predicting when the browser is ready.
  if (webcamRunning === true) {
    window.requestAnimationFrame(predictWebcam);
  }
}
function drawChart(item){
  var update = {
    x:  [[item.time],[item.time],[item.time]],
    y: [[item.p],[item.e],[item.m]]
    }
    console.log("item.p="+item.p);
    Plotly.extendTraces('gd', update, [0,1,2])
}
function initChart(){

  var trace1 = {
    x: [],
    y: [],
    mode: 'lines',
    name: 'Present',
    line: {
      color: '#80CAF6',
    }
  }
  
  var trace2 = {
    x: [],
    y: [],
    name: 'Look at',
    line: {color: '#DF56F1'}
  };
  var trace3 = {
    x: [],
    y: [],
    name: 'Silent',
    line: {color: 'rgb(128, 0, 128)'}
  };
  
  var layout = {
    xaxis: {
      type: 'date',
    },
    yaxis: {domain: [0.0,1]},
  }
  


  var data = [trace1,trace2,trace3];
//  var data = [trace1];
  
  Plotly.newPlot('gd', data, layout);
}
function drawBlendShapesExt(el, lastAttentionTime){
  let p = lastAttentionTime.p;
  let e = p*(1-lastAttentionTime.e);
  let m = p*(1-lastAttentionTime.m);
  let htmlMaker = '';
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">Present</span>
        <span class="blend-shapes-value" style="width: calc(${p * 100}% - 120px)">${(p).toFixed(4)}</span>
      </li>
    `;
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">Looks at</span>
        <span class="blend-shapes-value" style="width: calc(${e * 100}% - 120px)">${(+e).toFixed(4)}</span>
      </li>
    `;
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">Silent</span>
        <span class="blend-shapes-value" style="width: calc(${m * 100}% - 120px)">${(m).toFixed(4)}</span>
      </li>
    `;

  el.innerHTML = htmlMaker;

}
function drawBlendShapes(el, blendShapes) {
  return;
  if (!blendShapes.length) { return };
  let filter ={ ...mouthOpen, ...squint};
  let htmlMaker = '';
  blendShapes[0].categories.map((shape) => {
    if(filter[shape.displayName || shape.categoryName]){
    htmlMaker += `
      <li class="blend-shapes-item">
        <span class="blend-shapes-label">${shape.displayName || shape.categoryName}</span>
        <span class="blend-shapes-value" style="width: calc(${+shape.score * 100}% - 120px)">${(+shape.score).toFixed(4)}</span>
      </li>
    `;
    }
  });

  el.innerHTML = htmlMaker;
}

