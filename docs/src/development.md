```@meta
CollapsedDocStrings = true
```

# Development manual

### Contributing
If you are interested in contributing to the model, please contact the developers using the information on the Home page.

### Coding style
- Indentation uses 4 spaces, no tabs.
- Function names should be lowercase, with words separated by underscores .
- Lines should aim to have a length of no more than 92 characters.
- All functions should have docstrings, ideally with Arguments and Returns listed.
- More comments are always better, even if they seem redundant.
- Use type hinting where possible.
- Print statements should be made through the logger where possible.

### Code reference

```@autodocs
Modules = [
    Obliqua.solid0d, 
    Obliqua.solid1d, 
    Obliqua.solid1d_mush, 
    Obliqua.solid1d_relax, 
    Obliqua.solid1d_mush_relax, 
    Obliqua.solid1d.common,
    Obliqua.fluid0d,
    Obliqua.Hansen
]
Order   = [:function, :type]     
```