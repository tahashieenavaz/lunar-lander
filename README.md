# Lunar Lander in PyTorch

## Results

### First Attempts

<table align="center">
  <tr>
    <td><img src="./art/episode-0.gif" width="250"></td>
    <td><img src="./art/episode-1.gif" width="250"></td>
    <td><img src="./art/episode-2.gif" width="250"></td>
  </tr>
</table>

### Middle Attempts

<table align="center">
  <tr>
    <td><img src="./art/episode-249.gif" width="250"></td>
    <td><img src="./art/episode-250.gif" width="250"></td>
    <td><img src="./art/episode-251.gif" width="250"></td>
  </tr>
</table>

### Last Attempts

### Best Attempts

Following are the best attempts made by **the model** (epsilon < 0.1).

## Settings

<table>
    <tr>
        <td>#</td>
        <td>Variable</td>
        <td>Value</td>
        <td>Notes</td>
    </tr>
    <tr>
        <td>1</td>
        <td>Epsiodes</td>
        <td>500</td>
        <td>Total number of episodes.</td>
    </tr>
    <tr>
        <td>2</td>
        <td>$\gamma$</td>
        <td>0.95</td>
        <td>Future rewards importance.</td>
    </tr>
    <tr>
        <td>3</td>
        <td>$\epsilon$</td>
        <td>0.9</td>
        <td>Controls the trade-off between exploration and exploitation.</td>
    </tr>
    <tr>
        <td>4</td>
        <td>$\epsilon$ decay</td>
        <td>0.99104076</td>
        <td>Reaches the $\epsilon$ min in exactly 500 episodes.</td>
    </tr>
    <tr>
        <td>5</td>
        <td>$\tau$</td>
        <td>0.01</td>
        <td>Used in soft updates.</td>
    </tr>
    <tr>
        <td>6</td>
        <td>$\lambda$</td>
        <td>0.001</td>
        <td>Learning rate</td>
    </tr>
    <tr>
        <td>7</td>
        <td>$\theta$</td>
        <td>128</td>
        <td>Memory replay for consistent learning experience.</td>
    </tr>
</table>
