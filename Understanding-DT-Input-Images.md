# Understanding DT Input Images


|#| Max<br>Count| Name| Req'd| Strength via|Notes|Object|
| :-: | :-: | :-- | :-: | -- | -- | -- |
|1a|1|canvas/image|T2I: omitted <br> I2I: required|-<br>I2I Strength||`request.image`|
|1b|1|canvas/image mask|T2I: omitted <br> I2I: optional|no direct strength control| mask shape + mask blur/outset<br>control effect boundaries|`request.mask`|
|2|*n*<br>(model-dependent)|Moodboard|optional| Moodboard slider<br>(guidance weight)||`request.hints`<br>(hintType=`shuffle`;<br>repeated tensors)|
|3|*n*|ControlNets|optional|control weight<br>(and start/end range if exposed)|app commonly shows Depth,<br>Pose, Scribble, Color; API is not limited to only these 4|`request.hints` + `configuration.controls` (control model + mode/input override)<br><br>Depth: hintType=`depth`<br>Pose: hintType=`pose`<br> Scribble: hintType=`scribble`<br>Color/Colour: hintType=`color`|
|4a|1|Custom|optional|custom/control<br>guidance weight|custom is not a second base canvas image;<br>it is a custom guidance channel|`request.hints` (hintType=`custom`)|
|4b|1|Custom Mask|optional|no direct strength slider|defines where custom guidance<br>applies; effect shaped by mask geometry/blur settings|`request.mask`|