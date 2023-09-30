windows.dash_clientside = Object.assign({}, windows.dash_clientside, {
  clientside: {
    /**
     * @typedef {Array<[key: string]: Array<number> | string} DataArray
     * @param {DataArray} data 
     */
    open_financials_modal: function(data) {
      dialog = document.getElementById('dialog:stock-financials')
      dialog.showModal()

      if (!row) { return {}}

      return {
        data: data
      }
    }
  },
  /**
   * @param {number} n_clicks
   * @param {string} class_name
   * @returns {string}
   */
  close_financials_modal: function(n_clicks, class_name) {
    dialog = document.getElementById('dialog:stock-financials')
    if (n_clicks > 0) {dialog.close()}

    return class_name
  }
})