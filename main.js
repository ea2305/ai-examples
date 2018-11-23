/**
 * Point object
 * @author Elihu A. Cruz <elihuacruz@gmail.com>
 * @version 0.1.1
 */

// Variables
let pointers = [] // { x: '?', y: '?' } 

// Trained layer and bias
let m, b;

// Optimazer and learning rate
const learningRate = 0.01;
const optimizer = tf.train.sgd(learningRate);

/**
 * P5.js Main sketch configuration
 */
function setup () {
  // custom size canvas
  createCanvas(800, 800)

  // initialize m and b
  m = tf.variable(
    tf.scalar(random(1)) // generate start point
  )

  b = tf.variable(
    tf.scalar(random(1)) // randomize bias
  )
}

/**
 * Insert Point into list
 */
function mousePressed() {
  pointers.push(
    new Pointer(
      map(mouseX, 0, width, 0, 1), // x axis
      map(mouseY, 0, height, 1, 0), // y axis
      15 // size
    )
  )
}

/**
 * Main loop drawing
 */
function draw () {
  background('black') // Fill screen color

  if (pointers.length > 0) { // Dataset validator

    tf.tidy(() => { // Clean memory
      
      const listY = pointers.map(point => point.y)
      const listX = pointers.map(point => point.x)

      const ys = tf.tensor1d(listY) // Generate Y pointer tensor

      // Update Network
      optimizer.minimize(() => loss(
        networkOut(listX), // Network output tensor
        ys //
      ))
    })

  }

  for (let i = 0; i < pointers.length; i++) {
    // Show points
    pointers[i].showInFrame()
  }

  // Generate line
  const lineXs = [0,1]
  const tensorYs = tf.tidy(() => networkOut(lineXs)) // Clean up memory

  let x1 = map(lineXs[0], 0, 1, 0, width) // begin point X line
  let x2 = map(lineXs[1], 0, 1, 0, width) // end point X line

  // Convert tensor to Array
  let lineYs = tensorYs.dataSync()
  
  tensorYs.dispose() // Clean up memory

  let y1 = map(lineYs[0], 0, 1, height, 0) // begin point Y line
  let y2 = map(lineYs[1], 0, 1, height, 0) // end point Y line

  // Colors
  stroke('#6223A0')
  strokeWeight(2)
  
  // Draw line
  line(x1, y1, x2, y2)
  noStroke()
}

/**
 * Generate a output of network
 * @param {Array} vector : Pointer X/Y Axis
 */
function networkOut (vector) {
  const xs = tf.tensor1d(vector)
  
  const ys = xs.mul(m)
              .add(b)

  return ys
}

/**
 * Ajust network value
 * @param {Tensor} pred Prediction Network output
 * @param {Tensor} label Training value output
 */
function loss (pred, label) {
  return pred.sub(label).square().mean()
}
