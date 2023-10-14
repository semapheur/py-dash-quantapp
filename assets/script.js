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
     * @param {Object|Array} select
     * @param {string} dialog_id
     * @returns {Figure}
     */
    graph_modal: function(select, dialog_id) {
      if (select === undefined) { return select}
      open_modal(dialog_id)
      return select
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