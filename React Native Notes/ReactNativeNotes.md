# React-Native-Notes

Renders **Native** components instead of html components like react js <br>

Video tutorial : https://youtu.be/ZBCUegTZF7M?si=UvL6lcsjDGUANefT <br>
Repo: https://github.com/adrianhajdin/aora <br>

**<View>**: Container which uses flexbox
**<Text>**: Used instead of h1 or p

```jsx
import React from 'react'
import {View, Text} from 'react-native
const App () => {
return (
<View>
<Text> Hello World! </Text>
</View>
```

### TouchableOpacity

Interactive element that fades in opacity when touched (like a button)

```jsx
import { TouchableOpacity, Text } from 'react-native';
function MyButton(props) {
return (
<TouchableOpacity onPress={props.onPress}>
<Text>{props.label} </Text>
</TouchableOpacity>
```

### TouchableHighlight

Interactive element reacting in a unique way

### TouchableWithoutFeedback

Self explanatory

### ActivityIndicator

A spinner to indicate loading

### Button

Self explanatory

### FlatList

**Long** list of elements to be scrolled in (like the map function)

_Extra Features:_

- Optimized Scroll Performance
- Item Seperation

### ScrollView

It is like a magic box that can hold multiple components and views, providing a scrolling container for them. (overflow:scroll)

### SafeAreaView

To allow all content to be viewed regardless of screen

```jsx
import SafeAreaView from "react-native-safe-area-context";
```

#### react-native-safe-context

is used because some report SafeAreaView doesn'nt always work

### Image

Display an image

### ImgBackground

To display image as background

### Modal

Like a popup

```jsx
import { Modal, Text,View } from 'react—native';

const App = () =>(
<Modal
visible={true}
animationType='slide'
onRequestClose={()-> console.log("close")}
>
  <View style={{flex:1,justifyContent:'center',alignltems:'center'}}>
    <Text> This is a modal</Text>
  </View>
</Modal>
```

### Alert

A popup

```jsx
import {Modal, Text, View } from 'react-native';
const App ) (= )
  <Modal
  visible={true}
  animationType="slide"
  onRequestClose={() ⇒ console.log('Modal closed')}
    <View style={{ flex: 1, justifyContent: 'center', alignItems:
    'center' }}>
      <Text>This is a modal</Text>
    </View>
  </Modal>
);
export default App
```

### Switch

A toggle

```jsx
import { Switch, View } from 'react-native';
const App = () => {
  const [isEnabled, setIsEnabled] ▪ useState(false);
  const toggleSwitch ()setIsEnabled (previous State ⇒
  !previousState);
  return (
    <View>
      <Switch
        trackColor={{ false: '#767577', true: '#81b0ff' }}
        thumbColor={isEnabled ? '#f5dd4b': '#f4f3f4'}
        onValueChange (toggleSwitch)
        value {isEnabled}
      />
    </View>
  );
  };
export default App
```

### StatusBar

To control the appearance top of the screen

# Initialization

-c means clear all the previous cache

```cmd
npx create-expo-app AppName
cd AppName
npx expo install expo-router react-native-safe-area-context react-native-screens expo-linking expo-constants expo-status-bar
npx expo start -c
```

In app.json change {scheme:"your app name"}, since this is used to deep link expo and react native applications <br>
It is a technique to allow the app to be opened using a specific string directly from a url outside of the app (eg: email,webpage or another app)
**name:** Name that appears on the app's homescreen
**slug:** URL friendly version of the app name

```json
{
  "expo": {
    "name": "app",
    "slug": "app",
    "version": "1.0.0",
    "orientation": "portrait",
    "icon": "./assets/images/icon.png",
    "scheme": "myapp",
    "userInterfaceStyle": "automatic",
    "splash": {
      "image": "./assets/images/splash.png",
      "resizeMode": "contain",
      "backgroundColor": "#ffffff"
    },
    "ios": {
      "supportsTablet": true
    },
    "android": {
      "adaptiveIcon": {
        "foregroundImage": "./assets/images/adaptive-icon.png",
        "backgroundColor": "#ffffff"
      }
    },
    "web": {
      "bundler": "metro",
      "output": "static",
      "favicon": "./assets/images/favicon.png"
    },
    "plugins": ["expo-router"],
    "experiments": {
      "typedRoutes": true
    }
  }
}
```

## Tailwind

tailwind.config.js file <br> \*_: means all folders <br>
_: means all files

```js
/** @type {import('tailwindcss').Config} */
dule.exports = {
content: "./app/**/*.{js,jsx, ts, tsx}", "./components/**/*. {js, jsx, ts, tsx}"],
theme: {
extend: {},
},
plugins: [],
}
```

## Notes

- **Directories** are called _route groups_ and the **files\*** inside them _screens_
- \_layout is visible on all screens
- If we have a directory with parenthesis (directory name) -> it can have its own \_layout
- **Dynamic Route** is [fileName]
- Can not have

```js
// Wrong
useEffect(async () => {
  // logic
});

//Correct
useEffect(() => {
  asyncronousFunction = async () => {
    //Logic
  };
  asyncronousFunction();
});
```

### UseEffect Unexpected Behaviour

#### Problem

If the useEffect isnt getting called when the elements of the dependencies array have changed, could be due:
React may optimize unnecessary re-renders, especially if the `element` in the dependencies array is an object then React is technically "shallow equal" to its previous value (even though its internal content changes). This can cause the effect not to run again if React thinks the instructions prop hasn't really changed (due to shallow comparison).

#### Solution

Pass a new reference

```js
setVariable([...newVariableArray]);
setVariable({ ...newVariableObject });
```

## Shortcuts

**rnfes:** To get the boilerplate code
