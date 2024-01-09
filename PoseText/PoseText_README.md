**Pose Annotation Explainations:**

- Please check the table below, please make sure the same frame numbers after your preprocess.
- Each json file contains all frames annotations of one motion and goes by {frame_idx, annotations}. 

<div align="center">
<table cellspacing="0" cellpadding="0" bgcolor="#ffffff" border="0">
  <tr>
    <th align="center">Dataset</th>
    <th align="center">Clip Number</th>
    <th align="center">Frame Number</th>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>aist++</b></td>
    <td align="center">1470</td>
    <td align="center">340928</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>animation</b></td>
    <td align="center">329</td>
    <td align="center">38136</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>dance</b></td>
    <td align="center">163</td>
    <td align="center">36078</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>egobody</b></td>
    <td align="center">980</td>
    <td align="center">438956</td>
  </td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>fitness</b></td>
    <td align="center">16730</td>
    <td align="center">3584563</td>
  <tr></tr>
  <tr>
    <td align="center"><b>game_motion</b></td>
    <td align="center">10217</td>
    <td align="center">1120002</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>GRAB</b></td>
    <td align="center">1335</td>
    <td align="center">406264</td>
     </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>HAA500</b></td>
    <td align="center">5231</td>
    <td align="center">311592</td>
</td>
  </tr>
  <tr></tr>
  <tr>
    <td align="center"><b>humanml</b></td>
    <td align="center">26292</td>
    <td align="center">3579846</td>
  </tr>
  <tr></tr>
  <tr></tr>
  <tr>
    <td align="center"><b>humman</b></td>
    <td align="center">744</td>
    <td align="center">104981</td>
  </tr>
  <tr>
    <td align="center"><b>idea400</b></td>
    <td align="center">12513</td>
    <td align="center">2594858</td>
  </tr>
  <tr>
    <td align="center"><b>kungfu</b></td>
    <td align="center">1040</td>
    <td align="center">257764</td>
  </tr>
  <tr>
    <td align="center"><b>music</b></td>
    <td align="center">3565</td>
    <td align="center">876748</td>
  </tr>
  <tr>
    <td align="center"><b>perform</b></td>
    <td align="center">475</td>
    <td align="center">102522</td>
  </tr>
</table>
</div>


Our basic usage (only random sample one frame pose description):
```
with open(body_text_name + '.json', 'r') as body_f:
    body_dict = json.load(body_f)

with open(body_text_name + '.json', 'r') as hand_f:
    hand_dict = json.load(hand_f)

select_frame = random.randint(0, len(hand_dict)-1)
hand_frame_text, body_frame_text = hand_dict[str(select_frame)], body_dict[str(select_frame)]

```
