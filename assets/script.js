/**
 * 
 * @param {HTMLElement} el 
 * @param {boolean} withChildren 
 */
function removeAllEventListeners(el, withChildren = false) {
  if (withChildren) {
    el.parentNode.replaceChild(el.cloneNode(true), el)
  }
  else {
    const newEl = el.cloneNode(false)
    while (el.hasChildNodes()) newEl.appendChild(el.firstChild)
    el.parentNode.replaceChild(newEl, el)
  }
}

/**
 * @param {string} dialog_id 
 */
function open_dialog(dialog_id) {
  dialog = document.getElementById(dialog_id)
  dialog.showModal()

  dialog.addEventListener("click", e => {
    const dialogDimensions = dialog.getBoundingClientRect()
    if (
      e.clientX < dialogDimensions.left ||
      e.clientX > dialogDimensions.right ||
      e.clientY < dialogDimensions.top ||
      e.clientY > dialogDimensions.bottom
    ) {
      dialog.close()
      removeAllEventListeners(dialog, false)
    }
  })
}

/**
 * @param {string} dialog_id 
 */
function close_dialog(dialog_id) {
  dialog = document.getElementById(dialog_id)
  dialog.close()
  removeAllEventListeners(dialog, false)
}

/**
 * @param {string|Object} id 
 */
function check_id(id) {
  if (typeof id !== "object") { return id }

  sorted_keys = Object.keys(id).sort()
  const sorted_object = {}
  sorted_keys.forEach((key) => {
    sorted_object[key] = id[key]
  })
  return JSON.stringify(sorted_object)
}

window.dash_clientside = Object.assign({}, window.dash_clientside, {
  clientside: {
    /**
     * @param {Object} cell
     * @param {string|Object} dialog_id
     * @returns {Object}
     */
    dcf_factor_modal: function(cell, dialog_id) {
      if (cell === undefined || cell.colId != "factor") { return cell }

      dialog_id = check_id(dialog_id)
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
     * @param {Object|string} dialog_id
     * @returns {string}
     */
    open_modal: function(n_clicks, dialog_id) {
      dialog_id = check_id(dialog_id)
      if (n_clicks === undefined || n_clicks === 0) { return dialog_id }
      open_dialog(dialog_id)
      return dialog_id
    },
    /**
     * @param {number} n_clicks
     * @param {string} dialog_id
     * @returns {string}
     */
    close_modal: function(n_clicks, dialog_id) {
      dialog_id = check_id(dialog_id)
      if (n_clicks > 0) {close_dialog(dialog_id)}

      return dialog_id
    },
    /**
     * @param {number} open_stamp
     * @param {number} close_stamp
     * @param {string} class_name
     * @returns {string}
     */
    handle_modal: function(open_stamp, close_stamp, dialog_id) {
      dialog_id = check_id(dialog_id)

      if (open_stamp === undefined && close_stamp === undefined) {
        return dialog_id
      }

      if (close_stamp === undefined) {
        open_dialog(dialog_id)
      } else if (open_stamp === undefined) {
        close_dialog(dialog_id)
      } else if (open_stamp > close_stamp) {
        open_dialog(dialog_id)
      } else if (close_stamp > open_stamp) {
        close_dialog(dialog_id)
      }
      return dialog_id
    }
  }
})