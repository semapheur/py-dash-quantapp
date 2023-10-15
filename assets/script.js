/**
 * @param {string} dialog_id 
 */
function open_modal(dialog_id) {
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
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    /**
     * @param {Object} cell
     * @param {string} dialog_id
     * @returns {Object}
     */
    dcf_factor_modal: function(cell, dialog_id) {
      console.log(cell)
      if (cell === undefined || cell.colId != 'factor') { return cell }
      open_modal(dialog_id)
      return cell
    },
    /**
     * @param {Array<Object>} row
     * @param {string} dialog_id
     * @returns {Array<Object>}
     */
    row_select_modal: function(row, dialog_id) {
      if (row === undefined) { return row }
      open_modal(dialog_id)
      return row
    },
    /**
     * @param {number} n_clicks
     * @param {string} dialog_id
     * @returns {string}
     */
    modal: function(n_clicks, dialog_id) {
      if (n_clicks === 0) { return dialog_id }
      open_modal(dialog_id)
      return dialog_id
    },
    /**
     * @param {number} n_clicks
     * @param {string} class_name
     * @returns {string}
     */
    close_modal: function(n_clicks, dialog_id) {
      dialog = document.getElementById(dialog_id)
      if (n_clicks > 0) {dialog.close()}

      return dialog_id
    }
  }
})