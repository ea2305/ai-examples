/**
 * Point object
 * @author Elihu A. Cruz <elihuacruz@gmail.com>
 * @version 0.1.1
 */

// Variables
let pointers = [] // { x: '?', y: '?' } 

// Trained layer and bias
let a, b, c, d;

// Optimazer and learning rate
const learningRate = 0.05;
const optimizer = tf.train.adam(learningRate);

/**
 * P5.js Main sketch configuration
 */
function setup () {
  // custom size canvas
  createCanvas(800, 800)

  // initialize m and b
  a = tf.variable(
    tf.scalar(random(0, 1)) // generate start point
  )

  b = tf.variable(
    tf.scalar(random(0, 1)) // randomize bias
  )

  c = tf.variable(
    tf.scalar(random(0, 1)) // randomize bias
  )
  
  d = tf.variable(
    tf.scalar(random(0, 1)) // randomize bias
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
  let curveX = []
  for (let i = 0; i <= 1; i += 0.01)
    curveX.push(i)

  const tensorYs = tf.tidy(() => networkOut(curveX)) // Clean up memory
  // Convert tensor to Array
  let curveYs = tensorYs.dataSync()
    
  tensorYs.dispose() // Clean up memory

  // Colors
  beginShape()
  noFill()
  stroke('#6223A0')
  strokeWeight(2)

  for (let j = 0; j < curveX.length; j++) {
    const x = map(curveX[j], 0, 1, 0, width)
    const y = map(curveYs[j],0, 1, height, 0)
    vertex(x, y)
  }
  endShape()
  noStroke()

}

/**
 * Generate a output of network
 * @param {Array} vector : Pointer X/Y Axis
 */
function networkOut (vector) {
  const xs = tf.tensor1d(vector)
  
  // polynomial 3rd 
  // y = ax^3 + bx^2 + cx + d

  const ys = xs.pow(tf.scalar(3)).mul(a)
    .add(xs.square().mul(b))
    .add(xs.mul(c))
    .add(d)

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
