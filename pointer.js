class Pointer {
  /**
   * Initialize enviroment
   * @param {Number} x X axis position
   * @param {Number} y Y axis position
   * @param {Number} size Circle radius
   * @param {String} color Main Color
   */
  constructor (x = 0, y = 0, size = 25, color = '#F09898') {
    this.x = x
    this.y = y
    this.size = size
    this.color = color
  }

  /**
   * Display point on screen with P5.js
   */
  showInFrame () {
    fill(this.color) // custom color
    ellipse(
      // reposition coordinates
      map(this.x, 0, 1, 0, width), // x axis
      map(this.y, 1, 0, 1, height), // y axis
      this.size
    )
  }
}