var dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

dagfuncs.DccDropdown = class {
  init(params) {
    this.params = params
    this.ref = React.createRef()

    const setProps = (props) => {
      if (typeof props.value != 'undefined') {
        this.value = props.value
        delete params.colDef.suppressKeyboardEvent
        params.api.stopEditing()
        this.prevFocus.focus()
      }
    }

    this.eInput = document.createElement('div')

    ReactDOM.render(react.createElement(window.dash_core_components.Dropdown, {
      options: params.value,
      value: params.value,
      ref: this.ref,
      setProps,
      style: {
        width: params.column.actualWidth
      }
    }))
    // allows focus event
    this.eInput.tabIndex = '0' 

    // set editor value to value from the cell
    this.value = params.value
  }

  getGui() {
    return this.eInput
  }

  focusChild() {
    const clickEvent = new MouseEvent('mousedown', {
      view: window,
      bubbles: true
    })

    setTimeout(() => {
      // const inp = this.eInput.getElementsByClassName('Select-value')[0]
      const inp = this.eInput.getElementsByClassName('Select-arrow')[0]
      inp.tabIndex = '1'

      this.params.colDef.suppressKeyboardEvent = (params) => {
        const gridShouldDoNothing = params.stopEditing
        return gridShouldDoNothing
      }
      inp.dispatchEvent(clickEvent)
    }, 100)
  }

  afterGuiAttached() {
    this.prevFocus = document.activeElement
    this.eInput.addEventListener('focus', this.focusChild())
    this.eInput.focus()
  }

  getValue() {
    return this.value
  }

  destroy() {
    this.prevFocus.focus()
  }
}

distributions = {
  normal: {
    parameters: {
      mu: 'Mean',
      sigma: 'Scale'
    }
  },
  skewnormal: {
    parameters: {
      a: 'Skew',
      loc: 'Mean',
      scale: 'Scale'
    }
  },
  triangular: {
    parameters: {
      a: 'Min',
      m: 'Mode',
      b: 'Max'
    }
  },
  uniform: {
    parameters: {
      a: 'Min',
      b: 'Max'
    }
  }
}

dagfuncs.ParameterInput = class {
  init(params) {
    const phase = params.colDef.field.split(':')[0]

    this.eForm = document.createElement('form')
    this.eForm.className = 'w-full flex gap-1'

    const dist = params.data[`${phase}:distribution`].toLowerCase()
    
    const distParams = distributions[dist]['parameters']
    const width = `${(1 / Object.keys(distParams).length) * 100}%`
    this.eInputs = []
    for (const key in distParams) {
      const eInput = document.createElement('input')
      eInput.placeholder = distParams[key]
      eInput.type = 'numeric'
      eInput.style.width = width

      this.eForm.appendChild(eInput)

      this.eInputs.push(eInput)
    }
  }

  getGui() {
    return this.eForm
  }

  afterGuiAttached() {
    this.eForm.focus();
  }

  getValue() {
    const values = []

    for (const input of this.eInputs) {
      values.push(input.value)
    }

    return `(${values.join(',')})`;
  }
}