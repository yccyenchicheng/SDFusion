'use strict';


// we start with the TrieNode
function TrieNode(key) {
  // the "key" value will be the character in sequence
  this.key = key;
  
  // we keep a reference to parent
  this.parent = null;
  
  // we have hash of children
  this.children = {};
  
  // check to see if the node is at the end
  this.end = false;
}

// iterates through the parents to get the word.
// time complexity: O(k), k = word length
TrieNode.prototype.getWord = function() {
  var output = [];
  var node = this;
  
  while (node !== null) {
    output.unshift(node.key);
    node = node.parent;
  }
  
  return output.join('');
};

// -----------------------------------------

// we implement Trie with just a simple root with null value.
function Trie() {
  this.root = new TrieNode(null);
}

// inserts a word into the trie.
// time complexity: O(k), k = word length
Trie.prototype.insert = function(word) {
  var node = this.root; // we start at the root 😬
  
  // for every character in the word
  for(var i = 0; i < word.length; i++) {
    // check to see if character node exists in children.
    if (!node.children[word[i]]) {
      // if it doesn't exist, we then create it.
      node.children[word[i]] = new TrieNode(word[i]);
      
      // we also assign the parent to the child node.
      node.children[word[i]].parent = node;
    }
    
    // proceed to the next depth in the trie.
    node = node.children[word[i]];
    
    // finally, we check to see if it's the last word.
    if (i == word.length-1) {
      // if it is, we set the end flag to true.
      node.end = true;
    }
  }
};

// check if it contains a whole word.
// time complexity: O(k), k = word length
Trie.prototype.contains = function(word) {
  var node = this.root;
  
  // for every character in the word
  for(var i = 0; i < word.length; i++) {
    // check to see if character node exists in children.
    if (node.children[word[i]]) {
      // if it exists, proceed to the next depth of the trie.
      node = node.children[word[i]];
    } else {
      // doesn't exist, return false since it's not a valid word.
      return false;
    }
  }
  
  // we finished going through all the words, but is it a whole word?
  return node.end;
};

// returns every word with given prefix
// time complexity: O(p + n), p = prefix length, n = number of child paths
Trie.prototype.find = function(prefix) {
  var node = this.root;
  var output = [];
  
  // for every character in the prefix
  for(var i = 0; i < prefix.length; i++) {
    // make sure prefix actually has words
    if (node.children[prefix[i]]) {
      node = node.children[prefix[i]];
    } else {
      // there's none. just return it.
      return output;
    }
  }
  
  // recursively find all words in the node
  findAllWords(node, output);
  
  return output;
};

// recursive function to find all words in the given node.
function findAllWords(node, arr) {
  // base case, if node is at a word, push to output
  if (node.end) {
    arr.unshift(node.getWord());
  }
  
  // iterate through each children, call recursive findAllWords
  for (var child in node.children) {
    findAllWords(node.children[child], arr);
  }
}

(function() {
    // Format video selector for compositional prompts.
    // var captions = [
    //     "a 3D model of a squirrel",
    //     "a DSLR photo of a squirrel",
    //     "a DSLR photo of a squirrel wearing a colorful sweater",
    //     "a DSLR photo of a squirrel wearing a kimono",
    //     "a DSLR photo of a squirrel wearing a leather jacket ice skating",
    //     "a DSLR photo of a squirrel wearing a leather jacket on rollerblades",
    //     "a DSLR photo of a squirrel wearing a leather jacket playing the cello",
    //     "a DSLR photo of a squirrel wearing a leather jacket playing the electric bass",
    //     "a DSLR photo of a squirrel wearing a leather jacket reading a book",
    //     "a DSLR photo of a squirrel wearing a leather jacket reading the newspaper",
    //     "a DSLR photo of a squirrel wearing a leather jacket",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle in the desert",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle on a dirt road",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle on a road made of ice",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle on the beach",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle on the highway",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle on the moon",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a motorcycle through a field of lava",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a skateboard",
    //     "a DSLR photo of a squirrel wearing a leather jacket riding a unicycle",
    //     "a DSLR photo of a squirrel wearing a puffy jacket",
    //     "a DSLR photo of a squirrel wearing a suit of medieval armor",
    //     "a koala",
    //     "an orangutan",
    //     "a panda",
    //     "a porcelain squirrel",
    //     "a raccoon",
    //     "a squirrel carved out of wood",
    //     "a squirrel made out of marbles",
    //     "a squirrel"
    // ]

    // var journey = {
    //     "a DSLR photo of a squirrel": {
    //         // " wearing a colorful sweater": true,
    //         // " wearing a kimono": true,
    //         // " wearing a puffy jacket": true,
    //         // " wearing a suit of medieval armor": true,
    //         " wearing a leather jacket": {
    //             // "##self##": true,
    //             " riding a motorcycle": {
    //                 // "##self##": true,
    //                 " in the desert": true,
    //                 " through a field of lava": true,
    //                 " on a dirt road": true,
    //                 " on a road made of ice": true,
    //                 " on the beach": true,
    //                 " on the highway": true,
    //                 " on the moon": true
    //             },
    //             " riding a unicycle": true,
    //             " riding a skateboard": true,
    //             " ice skating": true,
    //             " on rollerblades": true,
    //             " playing the cello": true,
    //             " playing the electric bass": true,
    //             " reading a book": true,
    //             " reading the newspaper": true,
    //         }
    //     },
    //     // "a 3D model of a squirrel": true,
    //     "a squirrel": {
    //         // "##self##": true,
    //         " carved out of wood": true,
    //         " made out of marbles": true,
    //     },
    //     // "a koala": false,
    //     // "an orangutan": false,
    //     // "a panda": false,
    //     // "a porcelain squirrel": false,
    //     // "a raccoon": false
    // };

    let captions = ["a DSLR photo of a squirrel  chopping vegetables",
"a DSLR photo of a squirrel  dancing",
"a DSLR photo of a squirrel  eating a hamburger",
"a DSLR photo of a squirrel  playing the saxophone",
"a DSLR photo of a squirrel  reading a book",
"a DSLR photo of a squirrel  ",
"a DSLR photo of a squirrel  riding a motorcycle",
"a DSLR photo of a squirrel  riding a skateboard",
"a DSLR photo of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a kimono chopping vegetables",
"a DSLR photo of a squirrel wearing a kimono dancing",
"a DSLR photo of a squirrel wearing a kimono eating a hamburger",
"a DSLR photo of a squirrel wearing a kimono playing the saxophone",
"a DSLR photo of a squirrel wearing a kimono reading a book",
"a DSLR photo of a squirrel wearing a kimono ",
"a DSLR photo of a squirrel wearing a kimono riding a motorcycle",
"a DSLR photo of a squirrel wearing a kimono riding a skateboard",
"a DSLR photo of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a kimono wielding a katana",
"a DSLR photo of a squirrel wearing a medieval suit of armor chopping vegetables",
"a DSLR photo of a squirrel wearing a medieval suit of armor dancing",
"a DSLR photo of a squirrel wearing a medieval suit of armor eating a hamburger",
"a DSLR photo of a squirrel wearing a medieval suit of armor playing the saxophone",
"a DSLR photo of a squirrel wearing a medieval suit of armor reading a book",
"a DSLR photo of a squirrel wearing a medieval suit of armor ",
"a DSLR photo of a squirrel wearing a medieval suit of armor riding a motorcycle",
"a DSLR photo of a squirrel wearing a medieval suit of armor riding a skateboard",
"a DSLR photo of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a medieval suit of armor wielding a katana",
"a DSLR photo of a squirrel wearing an elegant ballgown chopping vegetables",
"a DSLR photo of a squirrel wearing an elegant ballgown dancing",
"a DSLR photo of a squirrel wearing an elegant ballgown eating a hamburger",
"a DSLR photo of a squirrel wearing an elegant ballgown playing the saxophone",
"a DSLR photo of a squirrel wearing an elegant ballgown reading a book",
"a DSLR photo of a squirrel wearing an elegant ballgown ",
"a DSLR photo of a squirrel wearing an elegant ballgown riding a motorcycle",
"a DSLR photo of a squirrel wearing an elegant ballgown riding a skateboard",
"a DSLR photo of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing an elegant ballgown wielding a katana",
"a DSLR photo of a squirrel wearing a purple hoodie chopping vegetables",
"a DSLR photo of a squirrel wearing a purple hoodie dancing",
"a DSLR photo of a squirrel wearing a purple hoodie eating a hamburger",
"a DSLR photo of a squirrel wearing a purple hoodie playing the saxophone",
"a DSLR photo of a squirrel wearing a purple hoodie reading a book",
"a DSLR photo of a squirrel wearing a purple hoodie ",
"a DSLR photo of a squirrel wearing a purple hoodie riding a motorcycle",
"a DSLR photo of a squirrel wearing a purple hoodie riding a skateboard",
"a DSLR photo of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"a DSLR photo of a squirrel wearing a purple hoodie wielding a katana",
"a DSLR photo of a squirrel  wielding a katana",
"a highly detailed metal sculpture of a squirrel  chopping vegetables",
"a highly detailed metal sculpture of a squirrel  dancing",
"a highly detailed metal sculpture of a squirrel  eating a hamburger",
"a highly detailed metal sculpture of a squirrel  playing the saxophone",
"a highly detailed metal sculpture of a squirrel  reading a book",
"a highly detailed metal sculpture of a squirrel  ",
"a highly detailed metal sculpture of a squirrel  riding a motorcycle",
"a highly detailed metal sculpture of a squirrel  riding a skateboard",
"a highly detailed metal sculpture of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a kimono chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a kimono dancing",
"a highly detailed metal sculpture of a squirrel wearing a kimono eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a kimono playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a kimono reading a book",
"a highly detailed metal sculpture of a squirrel wearing a kimono ",
"a highly detailed metal sculpture of a squirrel wearing a kimono riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a kimono riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a kimono wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor dancing",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor reading a book",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor ",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a medieval suit of armor wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown dancing",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown reading a book",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown ",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing an elegant ballgown wielding a katana",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie chopping vegetables",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie dancing",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie eating a hamburger",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie playing the saxophone",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie reading a book",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie ",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie riding a motorcycle",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie riding a skateboard",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"a highly detailed metal sculpture of a squirrel wearing a purple hoodie wielding a katana",
"a highly detailed metal sculpture of a squirrel  wielding a katana",
"an intricate wooden carving of a squirrel  chopping vegetables",
"an intricate wooden carving of a squirrel  dancing",
"an intricate wooden carving of a squirrel  eating a hamburger",
"an intricate wooden carving of a squirrel  playing the saxophone",
"an intricate wooden carving of a squirrel  reading a book",
"an intricate wooden carving of a squirrel  ",
"an intricate wooden carving of a squirrel  riding a motorcycle",
"an intricate wooden carving of a squirrel  riding a skateboard",
"an intricate wooden carving of a squirrel  sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a kimono chopping vegetables",
"an intricate wooden carving of a squirrel wearing a kimono dancing",
"an intricate wooden carving of a squirrel wearing a kimono eating a hamburger",
"an intricate wooden carving of a squirrel wearing a kimono playing the saxophone",
"an intricate wooden carving of a squirrel wearing a kimono reading a book",
"an intricate wooden carving of a squirrel wearing a kimono ",
"an intricate wooden carving of a squirrel wearing a kimono riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a kimono riding a skateboard",
"an intricate wooden carving of a squirrel wearing a kimono sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a kimono wielding a katana",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor chopping vegetables",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor dancing",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor eating a hamburger",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor playing the saxophone",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor reading a book",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor ",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor riding a skateboard",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a medieval suit of armor wielding a katana",
"an intricate wooden carving of a squirrel wearing an elegant ballgown chopping vegetables",
"an intricate wooden carving of a squirrel wearing an elegant ballgown dancing",
"an intricate wooden carving of a squirrel wearing an elegant ballgown eating a hamburger",
"an intricate wooden carving of a squirrel wearing an elegant ballgown playing the saxophone",
"an intricate wooden carving of a squirrel wearing an elegant ballgown reading a book",
"an intricate wooden carving of a squirrel wearing an elegant ballgown ",
"an intricate wooden carving of a squirrel wearing an elegant ballgown riding a motorcycle",
"an intricate wooden carving of a squirrel wearing an elegant ballgown riding a skateboard",
"an intricate wooden carving of a squirrel wearing an elegant ballgown sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing an elegant ballgown wielding a katana",
"an intricate wooden carving of a squirrel wearing a purple hoodie chopping vegetables",
"an intricate wooden carving of a squirrel wearing a purple hoodie dancing",
"an intricate wooden carving of a squirrel wearing a purple hoodie eating a hamburger",
"an intricate wooden carving of a squirrel wearing a purple hoodie playing the saxophone",
"an intricate wooden carving of a squirrel wearing a purple hoodie reading a book",
"an intricate wooden carving of a squirrel wearing a purple hoodie ",
"an intricate wooden carving of a squirrel wearing a purple hoodie riding a motorcycle",
"an intricate wooden carving of a squirrel wearing a purple hoodie riding a skateboard",
"an intricate wooden carving of a squirrel wearing a purple hoodie sitting at a pottery wheel shaping a clay bowl",
"an intricate wooden carving of a squirrel wearing a purple hoodie wielding a katana",
"an intricate wooden carving of a squirrel  wielding a katana"];

    // let imagen_pieces = [
    //     ["a DSLR photo of a squirrel", "an intricate wooden carving of a squirrel", "a highly detailed metal sculpture of a squirrel"],
    //     ["", "wearing a kimono", "wearing a medieval suit of armor", "wearing a purple hoodie", "wearing an elegant ballgown"],
    //     ["", "reading a book", "riding a motorcycle", "playing the saxophone", "chopping vegetables", "sitting at a pottery wheel shaping a clay bowl",
    //     "riding a skateboard", "wielding a katana", "eating a hamburger", "dancing"],
    // ];

    let journey = {
        "a DSLR photo of a squirrel": {
            "  ": true,
            " wearing a kimono": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a medieval suit of armor": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true, 
            },
            " wearing a purple hoodie": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a elegant ballgown": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
        },
        "a highly detailed metal sculpture of a squirrel": {
            "  ": true,
            " wearing a kimono": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a medieval suit of armor": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true, 
            },
            " wearing a purple hoodie": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a elegant ballgown": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
        },
        "an intricate wooden carving of a squirrel": {
            "  ": true,
            " wearing a kimono": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a medieval suit of armor": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true, 
            },
            " wearing a purple hoodie": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
            " wearing a elegant ballgown": {
                " ": true,
                " reading a book": true,
                " riding a motorcycle": true,
                " playing the saxophone": true,
                " chopping vegetables": true,
                " sitting at a pottery wheel shaping a clay bowl": true,
                " riding a skateboard": true,
                " wielding a katana": true,
                " eating a hamburger": true,
                " dancing": true,
            },
        }
    };

    // var trie = new Trie();
    // captions.forEach((caption) => {
    //     trie.insert(caption);
    // });
    // console.log('trie', trie);

    var N_LEVELS = 1;

    // Array.prototype.extend = function (other_array) {
    //     /* You should include a test to check whether other_array really is an array */
    //     other_array.forEach(function(v) {this.push(v)}, this);
    // }

    function getTagDOM(node) {
        var classes = "";
        // console.log('getTagDOM node', node);
        if (!node.parent || !node.visible)
            // Hide if this is the root node or the node is not marked visible.
            classes = classes + "hidden ";

        if (node.selected)
            classes = classes + "selected ";

        if (node.active)
            classes = classes + "active ";

        node.htmlNode.classList = classes;
        return node.htmlNode;
    };

    function getDOM(node, tags) {
        var tag = [getTagDOM(node), node.depth];
        tags.push(tag);
        // console.log('getDOM prefix', node.prefix, 'children:', node.children);
        for (var i = 0; i < node.children.length; i++) {
            var child = node.children[i];
            // console.log('getting dom for child', child.prefix)
            getDOM(child, tags);
        }
        return tags;
    };

    function treeMap(node, fn, maxdepth) {
        fn(node);
        if (maxdepth > 1) {
            for (var i = 0; i < node.children.length; i++)
                treeMap(node.children[i], fn, maxdepth-1);
        }
    }

    function deactivate(node) {
        node.active = false;
    }

    function unselect(node) {
        node.selected = false;
    }

    function hide(node) {
        node.visible = false;
    }

    function show(node) {
        node.visible = true;
    }

    function getOnSelect(node) {
        return function() {
            // Unselect and hide all nodes in the tree.
            var root = node;
            while (root.parent)
                root = root.parent;
            treeMap(root, deactivate, 999);
            treeMap(root, unselect, 999);
            treeMap(root, hide, 999);

            // Hide siblings using the parent pointer.
            // for (var i = 0; i < node.parent.children.length; i++) {
                // node.parent.children[i].visible = false;
                // node.parent.children[i].selected = false;
            // }

            // Reveal this node.
            node.selected = true;
            node.active = true;

            // Reveal parents and set active.
            var parent = node;
            while (parent) {
                parent.visible = true;
                parent.active = true;
                parent = parent.parent;
            }

            // Reveal siblings of parent.
            // if (node.parent)
            //     treeMap(node.parent, show, 2);

            // console.log('prefix', node.prefix, 'children len', node.children.length);
            // for (var j = 0; j < node.children.length; j++) {
            //     console.log('revealing child', node.children[j].prefix);
            //     node.children[j].visible = true;
            // }
            treeMap(node, show, 999);
            // console.log('selected', node);

            // Set video source.
            if (node.videoURL) {
                var video_src_el = document.getElementById("compositionalVideoSrc")
                // video_el.src = node.videoURL;
                video_src_el.setAttribute('data-src', node.videoURL);
            }

            redraw();
        }
    }

    class Node {
        constructor(parent, prefix, segment, depth) {
            this.depth = depth;
            this.parent = parent;
            this.prefix = prefix;
            this.segment = segment;
            if (captions.includes(this.prefix)) {
                this.videoURL = (
                    'https://pub-b1f092b6867f4495b8f149d222a3bffe.r2.dev/journey_sept28/full/' +
                    prefix.replaceAll(' ', '_') + '_rgbdn_hq_15000.mp4');
                // console.log(this.videoURL);
            }
            this.selected = false;
            // this.htmlNode = document.createElement("SPAN");
            // this.htmlNode.appendChild(document.createTextNode(this.segment));
            // this.htmlNode.onclick = this.select;
            this.children = [];

            this.htmlNode = document.createElement("SPAN");
            this.htmlNode.appendChild(document.createTextNode(this.segment.trim()));
            this.htmlNode.addEventListener("click", getOnSelect(this));
        }
    }

    var tree = new Node(undefined, "", "", 0);
    function addNodes(root, children_journey) {
        N_LEVELS = Math.max(N_LEVELS, root.depth);
        for (var segment in children_journey) {
            // console.log('adding segment', segment);
            var prefix = root.prefix + segment;
            N_LEVELS = Math.max(N_LEVELS, root.depth + 1);
            var node = new Node(root, prefix, segment, root.depth + 1);
            root.children.push(node);
            var grandchildJourney = children_journey[segment]
            if (!(grandchildJourney == true || grandchildJourney == false))
                addNodes(node, children_journey[segment]);
        }
    }
    addNodes(tree, journey);

    function redraw() {
        var scroll = document.documentElement.scrollTop;

        // Clear tags from the DOM.
        for (var depth = 0; depth < N_LEVELS; depth++) {
            var tags_el = document.getElementById("compositional_tags_depth_" + depth);
            // tags_el.innerHTML = "";
            while (tags_el.lastElementChild) {
                tags_el.removeChild(tags_el.lastElementChild);
            }
        }

        // Add in updated tags.
        var dom = [];
        getDOM(tree, dom);
        for (var i = 0; i < dom.length; i++) {
            var tags_el = document.getElementById("compositional_tags_depth_" + dom[i][1]);
            tags_el.appendChild(dom[i][0]);
        }

        // Reset scroll position to prevent jumping.
        document.documentElement.scrollTop = scroll;
    }

    // console.log(tree);
    // tree.htmlNode.click();
    tree.children[0].htmlNode.click();
    // redraw();
})();



(function() {
    // Click to load handlers for 3D meshes.
    document.querySelectorAll('button.loads-parent-model').forEach((button) => {
        button.addEventListener('click', () => {
            // button.classList = button.classList + " disappearing";
            let model = button.parentElement.parentElement;
            model.dismissPoster();
            button.classList = "btn btn-disabled";

            // model.addEventListener('poster-dismissed', () => {
            //     let originalTexture = model.model.materials[0].pbrMetallicRoughness.baseColorTexture;
            //     let originalBaseColor = model.model.materials[0].pbrMetallicRoughness.baseColorFactor;
            //     console.log('model load', model, originalTexture);

            //     let textureButton = model.querySelector('.toggles-parent-texture');
            //     console.log('texture button', textureButton);
            //     // if (originalTexture && textureButton) {
            //         let textureOn = true;
            //         textureButton.onclick = () => {
            //             if (textureOn) {
            //                 model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
            //                 model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor([1., 1., 1., 1.]);
            //                 textureOn = false;
            //                 console.log('toggle texture off');
            //             } else {
            //                 model.model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
            //                 model.model.materials[0].pbrMetallicRoughness.setBaseColorFactor(originalBaseColor);
            //                 textureOn = true;
            //                 console.log('toggle texture on');
            //             }
            //         };
            //     // }
            // });
        });
    });
    // document.querySelectorAll('button.toggles-parent-texture').forEach((button) => {
    //     let model = button.parentElement.parentElement;
    //     let originalTexture = model.materials[0].pbrMetallicRoughness.baseColorTexture;
    //     let textureOn = true;
    //     button.addEventListener('click', () => {
    //         if (textureOn) {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(null);
    //             textureOn = false;
    //         } else {
    //             model.materials[0].pbrMetallicRoughness.baseColorTexture.setTexture(originalTexture) ;
    //             textureOn = true;
    //         }
    //     });
    // });
})();
