# CSS-notes

## Remove underline from link <a>:
``` css
text-decoration: none;
```

## Remove bullet points from list:
``` css
/* <ul> */
list-style-type: none;
```

## Center a div:

### Absolutely/Fixed Positioned
``` css
left:50%;
transform: translateX(-50%);
top:50%;
transform: translateY(-50%);
```

### Relatively Positioned
``` css
/* Horizontally */
margin:0 auto;

/* Vertically */
/* Container Element */
display:flex;
align-items: center; // Vertical
justify-content: center; // Horizontal
```

### Psuedo Elements

::after - Selects a virtual element that is inserted after the content of the select element.
::before - Selects a virtual element that is inserted before the content of the select element.
::first-line - Selects the first line of text within the select element.
::first-letter - Selects the first letter of the content within the select element.
::selection - Selects the portion of the content that has been selected by the user.

```css
/* Styling the select element */
select {
    /* Your styles here */
}

/* Styling the options when hovered */
select option:hover {
    /* Your styles here */
}

/* Styling the select element when focused */
select:focus {
    /* Your styles here */
}

/* Styling the select element when active (clicked) */
select:active {
    /* Your styles here */
}

/* Styling the select element when it's disabled */
select:disabled {
    /* Your styles here */
}

```

##### Cannot add ::after on image:
The ::after pseudo-element is not displaying because it's being applied to the <img> element, which is a replaced element, and pseudo-elements like ::before and ::after don't work with replaced elements like images (<img>).

### Style Date, Datetime-local, Month, Week, Time 

```css
::-webkit-datetime-edit
::-webkit-datetime-edit-fields-wrapper
::-webkit-datetime-edit-text

::-webkit-datetime-edit-year-field
::-webkit-datetime-edit-month-field
::-webkit-datetime-edit-week-field
::-webkit-datetime-edit-day-field
::-webkit-datetime-edit-hour-field
::-webkit-datetime-edit-minute-field
::-webkit-datetime-edit-second-field
::-webkit-datetime-edit-millisecond-field
::-webkit-datetime-edit-ampm-field

::-webkit-inner-spin-button
::-webkit-calendar-picker-indicator
```        

