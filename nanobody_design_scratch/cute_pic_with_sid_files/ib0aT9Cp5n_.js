;/*FB_PKG_DELIM*/

__d("MAWDbAppMeta",[],(function(a,b,c,d,e,f){"use strict";a={allowSecurityAlert:"allowSecurityAlert",allowSecurityAlertForSelf:"allowSecurityAlertForSelf",deviceJid:"deviceJid",hasMigratedFromDexie:"hasMigratedFromDexie",hmacKey:"hmacKey",hotlikeSticker:"hotlikeSticker",isBackfilledForOccamadillo:"isBackfilledForOccamadillo",msgTypeVersion:"msgTypeVersion",restoreMigrationAttempts:"restoreMigrationAttempts",restoreToDexieMigrationComplete:"restoreToDexieMigrationComplete"};f.AppMetaKeysEnum=a}),66);
__d("MAWDbXMA",["I64"],(function(a,b,c,d,e,f,g){"use strict";var h;function a(a){return a}function b(a){return a}function c(a){return(h||(h=d("I64"))).to_float(a)}g.convertNumbersToXMAIds=a;g.convertNumberToXMAId=b;g.convertXMAId64ToXMAId=c}),98);
__d("MAWParseXMAFBConfig",["WAArmadilloXMA.pb","gkx"],(function(a,b,c,d,e,f,g){"use strict";b=new Set([(a=d("WAArmadilloXMA.pb")).EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_FEED_SHARE,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_SHORT,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_STORY_SHARE,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_STORY_REPLY,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_GAMING_CUSTOM_UPDATE,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.MSG_EXTERNAL_LINK_SHARE,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_AUDIO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_VIDEO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_MISSED_AUDIO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_MISSED_VIDEO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_GROUP_AUDIO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.RTC_GROUP_VIDEO_CALL,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_FEED_POST_PRIVATE_REPLY,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.MSG_HIGHLIGHTS_TAB_FRIEND_UPDATES_REPLY,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.MSG_HIGHLIGHTS_TAB_LOCAL_EVENT_REPLY,a.EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.FB_EVENT,c("gkx")("23929")?d("WAArmadilloXMA.pb").EXTENDED_CONTENT_MESSAGE_EXTENDED_CONTENT_TYPE.MSG_RECEIVER_FETCH:null].filter(Boolean));g.FB_SUPPORTED_TARGET_TYPES=b}),98);
__d("MAWLSDBEncryption",["I64","LSPlatformErrorChannel","MAWCryptoConsts","MAWKeychainCrypto","MAWKeychainNaClCrypto","MAWKeychainUtil","MWEARKeychainV3","QPLUserFlow","ReStoreDecryptionFailure","err","nullthrows","qpl","unrecoverableViolation"],(function(a,b,c,d,e,f,g){"use strict";var h,i=typeof window!=="undefined"?window:self;function j(a){a=atob(a);var b=a.length,c=new Uint8Array(b);for(var d=0;d<b;d++)c[d]=a.charCodeAt(d);return c.buffer}function k(a){var b="";a=new Uint8Array(a);var c=a.byteLength;for(var d=0;d<c;d++)b+=String.fromCharCode(a[d]);return i.btoa(b)}function l(a){return c("nullthrows")(JSON.stringify(a,function(a,b){if((h||(h=d("I64"))).isI64(b))return{type:"64",value:[b[0],b[1]]};else if(a===""||b===null||typeof b==="string"||typeof b==="number"||typeof b==="boolean"||Array.isArray(b))return b;else if(b===void 0)return{type:"u"};else if(b instanceof ArrayBuffer)return{type:"ab",value:k(b)};else if(b instanceof Uint8Array)return{l:b.byteLength,o:b.byteOffset,type:"ui8",value:k(b.buffer)};else if(b instanceof Map)return{e:l(Array.from(b.entries())),type:"m"};else if(b instanceof Set)return{type:"s",v:l(Array.from(b.values()))};else if(typeof b==="object")return{type:"o",value:l(b)};throw c("unrecoverableViolation")("Not supported","messenger_comet")}))}function m(a){return JSON.parse(a,function(a,b){if(a===""||b===null||typeof b==="string"||typeof b==="number"||typeof b==="boolean"||Array.isArray(b))return b;else if(typeof b==="object")if(b.type==="64"){b.value._tag="i64";return b.value}else if(b.type==="u")return void 0;else if(b.type==="ab")return j(b.value);else if(b.type==="ui8")return new Uint8Array(j(b.value),b.o,b.l);else if(b.type==="o")return m(b.value);else if(b.type==="m")return new Map(m(b.e));else if(b.type==="s")return new Set(m(b.v));throw c("unrecoverableViolation")("Not supported","messenger_comet")})}function a(a,b){try{var e=d("MWEARKeychainV3").getDbEncryptionKey(),f=e.key;e=e.version;e=d("MAWKeychainUtil").makeAAD(e,d("MAWCryptoConsts").CIPHER_ID);return n(f,c("nullthrows")(l(a)),e)}catch(a){throw c("err")(a.message+" - Encrypted LSDB was unable to encrypt an entity for table "+b)}}function b(a,b){try{var e=d("MAWKeychainCrypto").getKeyVersionFromCipherData(a),f=d("MWEARKeychainV3").getDbEncryptionKey(e);f=f.key;f=o(f,a,e);return c("nullthrows")(m(f))}catch(a){c("QPLUserFlow").addPoint(c("qpl")._(521481876,"1407"),"EBSM_HYDRATION_DECRYPTION_FAILURE_ON_"+b.toUpperCase());c("LSPlatformErrorChannel").emit(new(d("ReStoreDecryptionFailure").ReStoreDecryptionFailure)(a.message,b));throw new(d("ReStoreDecryptionFailure").ReStoreDecryptionFailure)(a.message,b)}}function n(a,b,c){b=new TextEncoder().encode(b).buffer;return d("MAWKeychainNaClCrypto").encryptTweetNaClArrayBuffer(a,new Uint8Array(b),c)}function o(a,b,c){a=d("MAWKeychainNaClCrypto").decryptTweetNaClArrayBuffer(a,b,c);return new TextDecoder().decode(a)}g.stringify=l;g.parse=m;g.encryptLSDBObj=a;g.decryptLSDBObj=b}),98);
__d("MAWSignalUtils",[],(function(a,b,c,d,e,f){"use strict";var g=null;function h(){if(g==null){g=[];for(var a=0;a<=255;a++){var b=a.toString(16).padStart(2,"0").toUpperCase();g.push(b)}}return g}function a(a){var b=h(),c=[];a.forEach(function(a){c.push(b[a])});return c.join(" ")}f.getHexRepresentation=a}),66);
__d("MWQPLMarkers",[],(function(a,b,c,d,e,f){"use strict";a={BEFORE_CREATE_OPTIMISTIC_MESSAGE:"before_create_optimistic_message",CREATE_OPTIMISTIC_MESSAGE_SUCCESS:"create_optimistic_message_success",CREATED_THREAD_FOR_XMA_SHARE:"created_thread_for_xma_share",FAIL_FORWARDING_MESSAGE:"fail_forwarding_message",FAILED_MEDIA_MISSING_THREAD:"failed_media_missing_thread",FAILED_MEDIA_MSGID_MISSING_IN_DB:"failed_media_msgid_missing_in_db",FAILED_MEDIA_UNSUPPORTED_TYPE:"failed_media_unsupported_type",FAILED_MESSAGE_MISSING_THREAD:"failed_message_missing_thread",FAILED_MISSING_ASSOCIATED_MESSAGE_IN_DB:"failed_missing_associated_message_in_db",FAILED_MISSING_THREAD_IN_DB:"failed_missing_thread_in_db",FAILED_MISSING_THREAD_IN_WRITING_XMA:"failed_missing_thread_in_writing_xma",FAILED_MISSING_XMA_MESSAGE_IN_DB:"failed_missing_xma_message_in_db",FAILED_TO_FETCH_BLOB_FOR_XMA_SHARE:"failed_to_fetch_blob_for_xma_share",FAILED_TO_FETCH_CHAT_JID_FOR_XMA_SHARE:"failed_to_fetch_chat_jid_for_xma_share",FAILED_TO_FETCH_QUERY_DATA_FOR_XMA_SHARE:"failed_to_fetch_query_data_for_xma_share",FAILED_TO_FETCH_THREAD_KEY_FOR_XMA_SHARE:"failed_to_fetch_thread_key_for_xma_share",FAILED_TO_PARSE_JSON_FOR_XMA_SHARE:"failed_to_parse_json_for_xma_share",FAILED_UPLOAD_MEDIA_TO_BACKEND:"failed_upload_media_to_backend",FETCHING_DEVICES_FOR_XMA_SHARE:"fetching_devices_for_xma_share",FORWARDED_MESSAGE_WRITTEN:"forwarded_message_written",GET_EXTERNAL_LINK_RESPONSE_SUCCESS:"get_external_link_response_success",MESSAGE_ALREADY_SENT_TO_BACKEND:"message_already_sent_to_backend",MISSING_XMA_DATA:"missing_xma_data",PREVIEW_DOES_NOT_EXIST:"preview_does_not_exist",PREVIEW_FAVICON_SPEC_READY:"preview_favicon_spec_ready",RECEIVED_FAILURE_SERVER_ACK:"received_failure_server_ack",RECEIVED_MEDIA_IN_WAJOB:"received_media_in_wajob",RECEIVED_MESSAGE_IN_WAJOB:"received_message_in_wajob",RECEIVED_SUCCESSFUL_SERVER_ACK:"received_successful_server_ack",RETURN_PRIVATE_XMA_DATA:"return_private_xma_data",RETURN_PUBLIC_XMA_DATA:"return_public_xma_data",SAVED_XMA_CONTENT_IN_DB:"saved_xma_content_in_db",SAVING_XMA_UPLOAD_RESULT_IN_DB:"saving_xma_upload_result_in_db",SEND_MESSAGE_TO_BACKEND_FAILED:"send_message_to_backend_failed",START_POST_SHARING:"start_post_sharing",START_STORY_REPLY:"start_story_reply",SUCCESSFUL_MESSAGE_SENT_TO_BACKEND:"successful_message_sent_to_backend",SUCCESSFUL_UPLOAD_MEDIA_TO_BACKEND:"successful_upload_media_to_backend",UPLOADING_MEDIA_TO_BACKEND:"uploading_media_to_backend",WRITING_MEDIA_TO_DB_FAILED:"writing_media_to_db_failed",WRITING_MESSAGE_TO_DB:"writing_message_to_db",WRITING_XMA_TO_DB:"writing_xma_to_db",XMA_DATA_FETCH_BEGIN:"xma_data_fetch_begin",XMA_DATA_FETCH_SUCCESS:"xma_data_fetch_success",XMA_DATA_QUERY_BEGIN:" xma_data_query_begin",XMA_DATA_QUERY_COMPLETE:"xma_data_query_complete"};f.MESSAGE_SEND_TO_SENT=a}),66);
__d("EBSMGating",["MAWWaitForBackendSetup","gkx"],(function(a,b,c,d,e,f,g){"use strict";function a(){return c("gkx")("24151")}function h(){return c("gkx")("1211")}function b(){return h()||d("MAWWaitForBackendSetup").isBackendSetupSuccessful()}g.isPersistedEBTableEnabled=a;g.isEBSMV2Enabled=h;g.isBackendSetupSuccessfulForEBSM=b}),98);
__d("decodeProtobuf",["WABinary","WAHasProperty","WAHex","WAProtoCompile","WAProtoConst","WAProtoUtils","WAProtoValidate"],(function(a,b,c,d,e,f,g){"use strict";function a(a,b){b=new(d("WABinary").Binary)(b);b=n(a,b,void 0,!1);d("WAProtoValidate").checkRequirements(a,b);return b}function b(a,b){b=new(d("WABinary").Binary)(b);b=n(a,b,void 0,!0);d("WAProtoValidate").checkRequirements(a,b);return b}function e(a){return c("WAHasProperty")(a,"$$unsafeUnknownFields")?a.$$unsafeUnknownFields:null}function h(a,b,c){if(a!==d("WAProtoUtils").typeToEncType(b))throw new Error("FormatError: "+c+" encoded with wire type "+a)}function i(a,b,c){switch(b){case d("WAProtoConst").TYPES.INT32:return j(c,-2147483648,2147483648,a,d("WABinary").parseInt64OrThrow);case d("WAProtoConst").TYPES.INT64:return c.readVarInt(k);case d("WAProtoConst").TYPES.UINT32:return j(c,0,4294967296,a,d("WABinary").parseUint64OrThrow);case d("WAProtoConst").TYPES.UINT64:return c.readVarInt(l);case d("WAProtoConst").TYPES.SINT32:b=j(c,0,4294967296,a,d("WABinary").parseInt64OrThrow);return b&1?~(b>>>1):b>>>1;case d("WAProtoConst").TYPES.SINT64:return c.readVarInt(m);case d("WAProtoConst").TYPES.BOOL:return!!j(c,0,2,a,d("WABinary").parseUint64OrThrow);case d("WAProtoConst").TYPES.ENUM:return c.readVarInt(d("WABinary").parseInt64OrThrow);case d("WAProtoConst").TYPES.FIXED64:return c.readLong(l,!0);case d("WAProtoConst").TYPES.SFIXED64:return c.readLong(k,!0);case d("WAProtoConst").TYPES.DOUBLE:return c.readFloat64(!0);case d("WAProtoConst").TYPES.STRING:return c.readString(c.readVarInt(d("WABinary").parseUint64OrThrow));case d("WAProtoConst").TYPES.BYTES:return c.readBuffer(c.readVarInt(d("WABinary").parseUint64OrThrow));case d("WAProtoConst").TYPES.FIXED32:return c.readUint32(!0);case d("WAProtoConst").TYPES.SFIXED32:return c.readInt32(!0);case d("WAProtoConst").TYPES.FLOAT:return c.readFloat32(!0)}}function j(a,b,c,d,e){a=a.readVarInt(e);if(a<b||a>=c)throw new Error("FormatError: "+d+" encoded with out-of-range value "+a);return a}function k(a,b){var c=d("WABinary").longFitsInDouble(!0,a,b);if(c){c=o(b);return a*4294967296+c}else{c=a<0;var e;c?e=b===0?-a:~a:e=a;a=c?-b:b;return d("WAHex").createHexLongFrom32Bits(e,a,c)}}function l(a,b){var c=d("WABinary").longFitsInDouble(!1,a,b);if(c){c=o(a);var e=o(b);return c*4294967296+e}else return d("WAHex").createHexLongFrom32Bits(a,b)}function m(a,b){var c=a>>>1;a=a<<31|b>>>1;b&1&&(c=~c,a=~a);return k(c,a)}function n(a,b,c,e){var f=d("WAProtoCompile").compileSpec(a),g=f.names,k=f.fields,l=f.types,m=f.meta,o=f.oneofToFields,p=f.fieldToOneof,q=f.reservedTags,r=f.reservedFields;f=a.internalDefaults;var s=c||babelHelpers["extends"]({},f)||{};s.$$unknownFieldCount=(a=c==null?void 0:c.$$unknownFieldCount)!=null?a:0;for(f=0;f<g.length;f++)l[f]&d("WAProtoConst").FLAGS.REPEATED&&(s[g[f]]=[]);var t=0;c=k.length>0;a=k[0];while(b.size()){f=j(b,0,4294967296,"field and enc type",d("WABinary").parseInt64OrThrow);var u=f&7,v=f>>>3;if(c&&v!==a){f=t;do++t===k.length&&(t=0),a=k[t];while(v!==a&&t!==f)}if(c&&v===a)(function(){var a=g[t],c=l[t];h(u,c,a);var f=c&d("WAProtoConst").TYPE_MASK,j=m[t];if(c&d("WAProtoConst").FLAGS.PACKED){var k=b.readVarInt(d("WABinary").parseUint64OrThrow);k=b.readBinary(k);while(k.size()){var w=i(a,f,k);(f!==d("WAProtoConst").TYPES.ENUM||j[w]||(j.cast==null?void 0:j.cast(w))!==void 0)&&s[a].push(w)}}else if(f===d("WAProtoConst").TYPES.MESSAGE){w=b.readVarInt(d("WABinary").parseUint64OrThrow);k=b.readBinary(w);if(c&d("WAProtoConst").FLAGS.REPEATED)s[a].push(n(j,k,void 0,e));else{w=s[a];s[a]=n(j,k,w,e)}}else{k=i(a,f,b);(f!==d("WAProtoConst").TYPES.ENUM||j[k]||(j.cast==null?void 0:j.cast(k))!==void 0)&&(c&d("WAProtoConst").FLAGS.REPEATED?s[a].push(k):s[a]=k)}w=p[a];w&&typeof s[a]!=="undefined"&&w.forEach(function(b){b=o[b].filter(function(b){return b!==a});b.forEach(function(a){delete s[a]})});(q[v]||r[a])&&delete s[a]})();else{s.$$unknownFieldCount++;if(e){s.$$unsafeUnknownFields||(s.$$unsafeUnknownFields={});f=void 0;switch(u){case d("WAProtoConst").ENC.VARINT:f=b.readVarInt(d("WABinary").parseInt64OrThrow);break;case d("WAProtoConst").ENC.BIT64:f=b.readBinary(8);break;case d("WAProtoConst").ENC.BINARY:f=b.readBinary(b.readVarInt(d("WABinary").parseUint64OrThrow));break;case d("WAProtoConst").ENC.BIT32:f=b.readBinary(4);break}s.$$unsafeUnknownFields[v]=f}else u===d("WAProtoConst").ENC.VARINT?b.readVarInt(d("WABinary").parseInt64OrThrow):u===d("WAProtoConst").ENC.BIT64?b.advance(8):u===d("WAProtoConst").ENC.BINARY?b.advance(b.readVarInt(d("WABinary").parseUint64OrThrow)):u===d("WAProtoConst").ENC.BIT32&&b.advance(4)}}return s}function o(a){return a>=0?a:4294967296+a}g.decodeProtobuf=a;g.decodeProtobufWithUnknowns=b;g.getUnknownFields=e}),98);
__d("WAPromiseTimeout",["Promise","WACustomError"],(function(a,b,c,d,e,f,g){"use strict";var h;a=function(a,c,e){var f=null,g=new(h||(h=b("Promise")))(function(a,b){f=setTimeout(function(){b(new(d("WACustomError").TimeoutError)(e)),clearTimeout(f)},c)});return h.race([a,g])["finally"](function(){clearTimeout(f)})};g.promiseTimeout=a}),98);
__d("WAByteArray",[],(function(a,b,c,d,e,f){"use strict";function a(a,b){b=b;var c=new Uint8Array(a);for(a=a-1;a>=0;a--)c[a]=b&255,b>>>=8;return c}function b(a){return a.buffer.slice(a.byteOffset,a.byteLength+a.byteOffset)}function c(a,b){if(!a||!b)return!1;a=new Uint8Array(a);b=new Uint8Array(b);var c=a.length,d=b.length;if(c!==d)return!1;for(d=0;d<c;d++)if(a[d]!==b[d])return!1;return!0}f.intToBytes=a;f.uint8ArrayToBuffer=b;f.compareArrayBuffer=c}),66);