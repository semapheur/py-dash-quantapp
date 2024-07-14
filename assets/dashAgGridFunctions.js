const dagfuncs = window.dashAgGridFunctions = window.dashAgGridFunctions || {};

dagfuncs.DccDropdown = class {
  init(params) {
    this.params = params
    this.ref = React.createRef()

    const setProps = (props) => {
      if (typeof props.value != "undefined") {
        this.value = props.value
        delete params.colDef.suppressKeyboardEvent
        params.api.stopEditing()
        this.prevFocus.focus()
      }
    }

    this.eInput = document.createElement("div")

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
    this.eInput.tabIndex = "0" 

    // set editor value to value from the cell
    this.value = params.value
  }
  getGui() {
    return this.eInput
  }
  focusChild() {
    const clickEvent = new MouseEvent("mousedown", {
      view: window,
      bubbles: true
    })

    setTimeout(() => {
      // const inp = this.eInput.getElementsByClassName("Select-value")[0]
      const inp = this.eInput.getElementsByClassName("Select-arrow")[0]
      inp.tabIndex = "1"

      this.params.colDef.suppressKeyboardEvent = (params) => {
        const gridShouldDoNothing = params.stopEditing
        return gridShouldDoNothing
      }
      inp.dispatchEvent(clickEvent)
    }, 100)
  }
  afterGuiAttached() {
    this.prevFocus = document.activeElement
    this.eInput.addEventListener("focus", this.focusChild())
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
    parameters: [
      "Mean",
      "Scale"
    ]
  },
  skewnormal: {
    parameters: [
      "Skew",
      "Mean",
      "Scale"
    ]
  },
  triangular: {
    parameters: [
      "Min",
      "Mode",
      "Max"
    ]
  },
  uniform: {
    parameters: [
      "Min",
      "Max"
    ]
  }
}
dagfuncs.ParameterInput = class {
  init(params) {
    const phase = params.colDef.field.split(":")[0]

    this.eForm = document.createElement("form")
    this.eForm.className = "w-full flex gap-1"

    const dist = params.data[`${phase}:distribution`].toLowerCase()
    
    const distParams = distributions[dist]["parameters"]
    const width = `${(1 / Object.keys(distParams).length) * 100}%`
    this.eInputs = []
    for (const i in distParams) {
      const eInput = document.createElement("input")
      eInput.value = params.value.split(", ")[i] ?? ""
      eInput.placeholder = distParams[i]
      eInput.type = "numeric"
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
    return values.join(", ");
  }
}
dagfuncs.NumberInput = class {
  init(params) {
    this.eInput = document.createElement("input")
    this.eInput.value = params.value
    this.eInput.type = "number"
    this.eInput.min = params?.min,
    this.eInput.max = params?.max,
    this.eInput.step = params.step || "any";
    this.eInput.placeholder =  params.placeholder || "";
  }
  getGui() {
    return this.eInput
  }
  afterGuiAttached() {
    this.eInput.focus();
    this.eInput.select();
  }
  getValue() {
    return this.eInput.value;
  }
}