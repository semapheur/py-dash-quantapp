window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    /**
     * @typedef {Object} Trace
     * @property {Array<string>} x
     * @property {Array<float>} y
     * @property {string} mode
     * 
     * @typedef {Object} Data
     * @property {string} title
     * @property {Array<Trace>} data
     * 
     * @param {Data} data
     * @param {string} dialog_id
     */
    open_financials_modal: function(data, dialog_id) {
      if (data === undefined || (
        typeof data == 'object' && Object.keys(data).length === 0)
      ) { return {} }
  
      dialog = document.getElementById(dialog_id)
      dialog.showModal()

      dialog.addEventListener('click', e => {
        const dialogDimensions = dialog.getBoundingClientRect()
        if (
          e.clientX < dialogDimensions.left ||
          e.clientX > dialogDimensions.right ||
          e.clientY < dialogDimensions.top ||
          e.clientY > dialogDimensions.bottom
        ) {
          dialog.close()
        }
      })
  
      return {
        data: data.data,
        layout: {
          title: data.title
        }
      }
    },
    /**
     * @param {number} n_clicks
     * @param {string} class_name
     * @returns {string}
     */
    close_financials_modal: function(n_clicks, dialog_id) {
      dialog = document.getElementById(dialog_id)
      if (n_clicks > 0) {dialog.close()}

      return dialog_id
    }
  }
})